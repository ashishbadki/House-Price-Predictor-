"""
check_deployment.py — Phase 26: run this BEFORE you deploy.

Deployment failures are slow to diagnose. You push, wait three minutes for a
build, read a log, guess, push again. Each round trip costs five minutes.

Every check here catches a problem that would otherwise cost you one of
those rounds. Run it locally in two seconds instead.

THE FAILURE THIS SCRIPT MOSTLY EXISTS FOR
-----------------------------------------
A `.joblib` file is a pickle of live Python objects. It stores the class
paths and the internal attribute layout of the scikit-learn version that
created it.

Load it under a different scikit-learn and one of three things happens:

  best   it works, with an InconsistentVersionWarning
  common it loads but an internal attribute is missing, and you get a
         confusing AttributeError deep inside sklearn on the first predict
  worst  it loads silently and predicts subtly different numbers

The third is the dangerous one. Nothing errors. The app just becomes wrong.

The fix is boring and complete: pin the exact versions in requirements.txt.
This script reads what the model was saved with (from model_card.json),
compares it to what is installed, and writes the pinned file for you.

Usage:
    python check_deployment.py
    python check_deployment.py --write-requirements
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app/app.py"
MODEL = ROOT / "models/house_price_model.joblib"
CARD = ROOT / "models/model_card.json"
LOCALITIES = ROOT / "models/locality_reference.json"
REQS = ROOT / "requirements.txt"

# Streamlit Community Cloud reads only ONE dependency file, in this priority
# order. If two exist, the later ones are ignored -- silently. A leftover
# Pipfile from an experiment will quietly override requirements.txt.
DEPENDENCY_FILES = ["uv.lock", "Pipfile", "environment.yml",
                    "requirements.txt", "pyproject.toml"]

# Only what the app imports at runtime. Anything else is dead weight that
# Community Cloud reinstalls on every cold start.
RUNTIME_PACKAGES = ["pandas", "numpy", "scikit-learn", "joblib", "streamlit"]

# Packages that must NOT be in requirements.txt -- they belong in
# requirements-dev.txt.
DEV_ONLY = ["pytest", "jupyterlab", "jupyter", "notebook", "ipykernel",
            "matplotlib", "seaborn", "xgboost", "lightgbm", "scipy"]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")
    return ok


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=20).stdout
    except Exception:
        return ""


def installed_version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


# ---------------------------------------------------------------------------

def check_files() -> None:
    print("\n1. FILES THE DEPLOYED APP NEEDS")
    check("app/app.py exists", APP.exists())
    check("models/house_price_model.joblib exists", MODEL.exists())
    check("models/locality_reference.json exists", LOCALITIES.exists())
    check("models/model_card.json exists", CARD.exists())

    if MODEL.exists():
        mb = MODEL.stat().st_size / 1024**2
        # GitHub hard-refuses over 100 MB. Over ~25 MB it also warns, and
        # every clone gets slower.
        check(f"model is a sensible size ({mb:.1f} MB)", mb < 90,
              "" if mb < 25 else "over 25 MB: GitHub will warn on push")


def check_git() -> None:
    print("\n2. WHAT IS ACTUALLY IN GIT")
    tracked = git("ls-files").splitlines()
    if not tracked:
        check("git repository initialised", False, "run `git init` first")
        return

    # The single most common deployment failure after version mismatch: the
    # model is gitignored, so the cloud clone has no model to load.
    check("model is committed", "models/house_price_model.joblib" in tracked,
          "" if "models/house_price_model.joblib" in tracked else
          "The cloud only sees committed files. Check .gitignore.")
    check("locality lookup is committed",
          "models/locality_reference.json" in tracked)

    data = [f for f in tracked if f.startswith(("data/raw/", "data/processed/"))
            and not f.endswith(".gitkeep")]
    check("no data files committed", not data,
          f"remove: {data[:3]}" if data else "")

    secrets = [f for f in tracked
               if f.endswith((".env", ".key", "secrets.toml", "secrets.json"))]
    check("no secret files committed", not secrets, str(secrets[:3]))

    dirty = git("status", "--porcelain").strip()
    check("everything is committed", not dirty,
          "uncommitted changes will not be deployed" if dirty else "")


def check_dependency_files() -> None:
    print("\n3. DEPENDENCY FILES")
    found = [f for f in DEPENDENCY_FILES if (ROOT / f).exists()]
    check("a dependency file exists", bool(found), f"found: {found}")
    # Community Cloud uses the FIRST match in its priority order and ignores
    # the rest, without saying so.
    check("only one dependency file", len(found) <= 1,
          f"Community Cloud will use ONLY '{found[0]}' and ignore {found[1:]}"
          if len(found) > 1 else "")

    if not REQS.exists():
        return
    lines = [l.split("#")[0].strip() for l in REQS.read_text().splitlines()]
    lines = [l for l in lines if l]
    names = [l.split("=")[0].split(">")[0].split("<")[0].split("[")[0].strip()
             .lower() for l in lines]

    for pkg in RUNTIME_PACKAGES:
        check(f"requirements.txt lists {pkg}", pkg.lower() in names)

    strays = sorted(set(names) & {d.lower() for d in DEV_ONLY})
    check("no dev-only packages in requirements.txt", not strays,
          f"move to requirements-dev.txt: {strays}" if strays else
          "keeps cold starts fast")

    unpinned = [l for l in lines if "==" not in l and not l.startswith("-")]
    check("every package is pinned with ==", not unpinned,
          f"unpinned: {unpinned[:4]}  -> run with --write-requirements"
          if unpinned else "")


def check_versions() -> None:
    print("\n4. VERSION MATCH  (the one that silently breaks predictions)")
    if not CARD.exists():
        check("model card present", False, "run src/train_final.py")
        return
    saved = json.loads(CARD.read_text()).get("versions", {})
    if not saved:
        check("model card records versions", False)
        return

    pkg_names = {"sklearn": "scikit-learn", "numpy": "numpy",
                 "pandas": "pandas", "python": None}
    for key, saved_version in saved.items():
        if key == "python":
            now = ".".join(str(x) for x in sys.version_info[:3])
            ok = now.split(".")[:2] == saved_version.split(".")[:2]
            check(f"python matches ({saved_version} saved, {now} here)", ok)
            continue
        now = installed_version(pkg_names[key])
        ok = now == saved_version
        check(f"{key} matches ({saved_version} saved, {now} here)", ok,
              "" if ok else "the model was pickled by a different version")


def check_model_loads() -> None:
    print("\n5. DOES THE MODEL ACTUALLY WORK")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from predict import list_localities, predict_price
            localities = list_localities()
            version_warnings = [w for w in caught
                                if "version" in str(w.message).lower()]
        check("predict module imports", True)
        check(f"locality list loads ({len(localities)} localities)",
              len(localities) > 50)
        check("no version warnings while loading", not version_warnings,
              str(version_warnings[0].message)[:110] if version_warnings else "")

        r = predict_price(locality=localities[0], area_sqft=900,
                          bedrooms=2, bathrooms=2)
        check(f"a real prediction works ({r['estimate_text']})",
              r["estimate"] > 0)

        from predict import InvalidInput
        try:
            predict_price(locality=localities[0], area_sqft=-1,
                          bedrooms=2, bathrooms=2)
            check("invalid input is rejected", False, "negative area accepted")
        except InvalidInput:
            check("invalid input is rejected", True)
    except Exception as e:
        check("predict module works", False, f"{type(e).__name__}: {e}")


def write_pinned_requirements() -> None:
    """Write requirements.txt with the exact versions installed right now.

    Exact pins are the whole point. `pandas>=2.0` lets the cloud install
    pandas 3.1 next month and break an app you have not touched.
    """
    lines = ["# Runtime dependencies for the deployed Streamlit app.",
             "#",
             "# EXACT pins, generated by check_deployment.py.",
             "# These match the environment the model was pickled in. A",
             "# different scikit-learn can load the pickle and predict",
             "# subtly different numbers with no error at all.",
             "#",
             "# Regenerate after retraining:",
             "#     python check_deployment.py --write-requirements",
             ""]
    missing = []
    for pkg in RUNTIME_PACKAGES:
        v = installed_version(pkg)
        if v:
            lines.append(f"{pkg}=={v}")
        else:
            missing.append(pkg)
    REQS.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {REQS} with exact pins:")
    for line in lines:
        if "==" in line and not line.startswith("#"):
            print(f"  {line}")
    if missing:
        print(f"  WARNING: not installed here, so not pinned: {missing}")
    print("\nCommit it, then redeploy.")


def main() -> int:
    if "--write-requirements" in sys.argv:
        write_pinned_requirements()
        return 0

    print("=" * 70)
    print("PRE-DEPLOYMENT CHECK")
    print("=" * 70)
    check_files()
    check_git()
    check_dependency_files()
    check_versions()
    check_model_loads()

    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 70)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("\nFix these before deploying:")
        for name, _, detail in failed:
            print(f"  - {name}")
            if detail:
                print(f"      {detail}")
        print("\nIf the only failures are about pinning, run:")
        print("    python check_deployment.py --write-requirements")
    else:
        print("\nReady to deploy. Push to GitHub, then:")
        print("  share.streamlit.io -> Create app -> pick your repo")
        print("  Main file path: app/app.py")
        print(f"  Python version: {'.'.join(str(x) for x in sys.version_info[:2])}"
              "  (set it under Advanced settings)")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
