"""
verify_setup.py — Phase 1 checkpoint script.

Purpose: prove that the environment is correctly built BEFORE we write any
real code. Debugging a broken environment while also debugging new code is
one of the most common ways beginners lose hours.

Run it with:   python verify_setup.py
"""

import sys
from importlib import import_module
from pathlib import Path

# The libraries we declared in requirements.txt, mapped to the name you
# actually type in an `import` statement (they are not always the same:
# you pip-install "scikit-learn" but you import "sklearn").
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "streamlit": "streamlit",
    "pytest": "pytest",
}

EXPECTED_DIRS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "app",
    "tests",
]


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    print(f"Python {major}.{minor} at {sys.executable}")
    ok = (major, minor) >= (3, 9)
    if not ok:
        print("  FAIL: Python 3.9 or newer is required.")
    return ok


def check_packages() -> bool:
    all_ok = True
    for pip_name, import_name in REQUIRED.items():
        try:
            module = import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  OK   {pip_name:<15} {version}")
        except ImportError:
            print(f"  FAIL {pip_name:<15} not installed")
            all_ok = False
    return all_ok


def check_folders() -> bool:
    root = Path(__file__).parent
    all_ok = True
    for folder in EXPECTED_DIRS:
        if (root / folder).is_dir():
            print(f"  OK   {folder}")
        else:
            print(f"  FAIL {folder} missing")
            all_ok = False
    return all_ok


def main() -> int:
    print("=" * 55)
    print("PHASE 1 SETUP VERIFICATION")
    print("=" * 55)

    print("\n[1] Python interpreter")
    python_ok = check_python()

    print("\n[2] Required packages")
    packages_ok = check_packages()

    print("\n[3] Project folders")
    folders_ok = check_folders()

    print("\n" + "=" * 55)
    if python_ok and packages_ok and folders_ok:
        print("ALL CHECKS PASSED — Phase 1 complete.")
        return 0

    print("SOME CHECKS FAILED — see the FAIL lines above.")
    return 1


if __name__ == "__main__":
    # sys.exit sets the process exit code: 0 = success, 1 = failure.
    # This is the convention that CI tools and scripts rely on.
    raise SystemExit(main())
