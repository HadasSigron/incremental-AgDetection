# tests/conftest.py
import os

# מניעת פתיחת חלונות GUI
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# דגל לוגי לריצה בטסטים (אם יש קוד ב-ui/ שמגיב לסביבה)
os.environ.setdefault("AGVISION_TEST", "1")
