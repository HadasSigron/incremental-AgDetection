# core/datasets/classification_folder.py
"""
Classification dataset adapter (ImageFolder-like) with optional train/val/test splits.

Supported layouts:

1) Split layout:
   root/
     train/<classes>/*
     val/<classes>/*
     [test/<classes>/*]        # optional

2) Train/Test only (no val):
   root/
     train/<classes>/*
     test/<classes>/*
   -> We map `test` to evaluation split (`val`) automatically.

3) Flat layout (single folder of classes):
   root/
     <ClassA>/*, <ClassB>/*, ...
   -> Treated as a single evaluation split (`val`).

Config fields (from cfg.dataset):
    root: str (required)
    train_dir: str | None (relative to root or absolute)
    val_dir:   str | None
    test_dir:  str | None
    extensions: list[str] | None

Sample dict:
    {"id": int, "path": str, "label": int | None}

Registered aliases:
    - classification_folder
    - classification
    - imagefolder
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

from .base import BaseAdapter

_IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


@dataclass
class _SplitView:
    root: Path                     # absolute path to split root (contains class subfolders)
    class_names: List[str]         # sorted class names for this split
    samples: List[Dict[str, Any]]  # sample dicts for this split


class ClassificationFolderAdapter(BaseAdapter):
    """
    ImageFolder-like adapter with optional splits.
    Normalizes class order across labeled splits and guarantees stable ids.
    """

    def __init__(
        self,
        root: Path,
        train_dir: Optional[Path],
        val_dir: Optional[Path],
        test_dir: Optional[Path],
        extensions: List[str],
    ) -> None:
        self._root = root.resolve()
        self._exts = [e.lower() for e in (extensions or _IMG_EXTS_DEFAULT)]

        # Build split views (if provided)
        self._split_train = self._scan_split(train_dir) if train_dir else None
        self._split_val   = self._scan_split(val_dir) if val_dir else None
        self._split_test  = self._scan_split(test_dir, labeled=False) if test_dir else None

        # --- Auto-detect common split names if nothing specified in YAML ---
        if not any([self._split_train, self._split_val, self._split_test]):
            auto_train = self._root / "train"
            auto_val   = self._root / "val"
            auto_test  = self._root / "test"

            if auto_train.exists() and auto_val.exists():
                self._split_train = self._scan_split(auto_train)
                self._split_val   = self._scan_split(auto_val)
                if auto_test.exists():
                    self._split_test = self._scan_split(auto_test, labeled=False)
            elif auto_train.exists() and auto_test.exists():
                # No explicit val: use test as evaluation split.
                self._split_train = self._scan_split(auto_train)
                self._split_val   = self._scan_split(auto_test)
            else:
                # Flat layout: treat root as a single eval split.
                self._split_val = self._scan_split(self._root)

        # Build unified class list from labeled splits
        labeled_sets = []
        for s in (self._split_train, self._split_val):
            if s:
                labeled_sets.append(set(s.class_names))
        unified = sorted(set().union(*labeled_sets)) if labeled_sets else []
        self._class_names: List[str] = unified
        self._cls2idx: Dict[str, int] = {c: i for i, c in enumerate(self._class_names)}

        # Remap labels in labeled splits to unified indices
        for s in (self._split_train, self._split_val):
            if not s:
                continue
            for smp in s.samples:
                cname = smp.pop("_cname")  # carried during scan
                smp["label"] = self._cls2idx[cname]

        # Assign stable consecutive ids across all splits
        running_id = 1
        for s in (self._split_train, self._split_val, self._split_test):
            if not s:
                continue
            for smp in s.samples:
                smp["id"] = running_id
                running_id += 1

    # -------------------- factory hook --------------------
    @classmethod
    def from_config(cls, cfg) -> "ClassificationFolderAdapter":
        ds = cfg.dataset
        root = Path(ds.root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"[classification] dataset.root not found: {root}")

        def _abs(p: Optional[str]) -> Optional[Path]:
            if not p:
                return None
            pp = Path(p)
            return (root / p).resolve() if not pp.is_absolute() else pp.resolve()

        train_dir = _abs(getattr(ds, "train_dir", None))
        val_dir   = _abs(getattr(ds, "val_dir", None))
        test_dir  = _abs(getattr(ds, "test_dir", None))
        exts      = getattr(ds, "extensions", None) or _IMG_EXTS_DEFAULT

        return cls(root=root, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, extensions=exts)

    # -------------------- API expected by runners --------------------
    def class_names(self) -> List[str]:
        return self._class_names

    def num_classes(self) -> int:
        return len(self._class_names)

    def split_exists(self, name: str) -> bool:
        name = (name or "").lower()
        if name == "train":
            return self._split_train is not None
        if name == "val":
            return self._split_val is not None
        if name == "test":
            return self._split_test is not None
        return False

    def iter_samples(self, split: str = "val") -> Iterable[Dict[str, Any]]:
        s = self._get_split(split)
        for smp in s.samples:
            yield smp

    def image_count(self, split: str = "val") -> int:
        return len(self._get_split(split).samples)

    # 👉 aliases for workers that expect different names
    def size(self, split: str = "val") -> int:
        """Alias to image_count for compatibility with workers that expect `size()`."""
        return self.image_count(split)

    def __len__(self) -> int:
        """Default length equals the evaluation split size."""
        return self.image_count("val")

    def describe(self) -> Dict[str, Any]:
        """Rich info for UI logs."""
        return {
            "root": str(self._root),
            "splits": {
                "train": str(self._split_train.root) if self._split_train else None,
                "val":   str(self._split_val.root) if self._split_val else None,
                "test":  str(self._split_test.root) if self._split_test else None,
            },
            "num_classes": self.num_classes(),
            "class_names": list(self._class_names),
            "counts": {
                "train": self.image_count("train") if self._split_train else 0,
                "val":   self.image_count("val")   if self._split_val   else 0,
                "test":  self.image_count("test")  if self._split_test  else 0,
            },
        }

    # -------------------- internals --------------------
    def _get_split(self, split: str) -> _SplitView:
        key = (split or "val").lower()
        s = {
            "train": self._split_train,
            "val":   self._split_val,
            "test":  self._split_test,
        }.get(key, None)
        if s is None:
            # Default to val if requested split is missing; this is safe for evaluation.
            if self._split_val is not None:
                return self._split_val
            raise ValueError(f"Requested split '{split}' is missing and no 'val' split is available.")
        return s

    def _scan_split(self, split_root: Path, labeled: bool = True) -> _SplitView:
        split_root = split_root.resolve()
        if not split_root.exists():
            raise FileNotFoundError(f"[classification] split path not found: {split_root}")

        # class directories are direct subfolders of the split root
        class_dirs = [d for d in split_root.iterdir() if d.is_dir()]
        class_dirs.sort(key=lambda p: p.name)

        class_names = [d.name for d in class_dirs]
        samples: List[Dict[str, Any]] = []

        for cdir in class_dirs:
            cname = cdir.name
            files = sorted([p for p in cdir.rglob("*") if p.suffix.lower() in self._exts])
            for p in files:
                samples.append({
                    "id": -1,                                 # assigned later
                    "path": str(p.resolve()),
                    "label": None if not labeled else -1,     # remapped later
                    "_cname": cname,                          # carried to remap to unified index
                })

        return _SplitView(root=split_root, class_names=class_names, samples=samples)
