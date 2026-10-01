# students/

Put enrolled students here. Two layouts work, and you can mix them:

```
students/
├── Ali Abbas/          ← folder per student (recommended)
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
└── Sara Khan.jpg       ← or a single photo named after the student
```

Tips for accurate recognition:

- 3–5 photos per person beats one. Vary the angle and lighting slightly.
- Clear, front-facing, one person per photo, face at least ~150 px wide.
- Easiest way: `python attendance.py enroll "Ali Abbas"` captures them from the webcam.

Everything in this folder except this README is git-ignored, so photos never end up on GitHub.
