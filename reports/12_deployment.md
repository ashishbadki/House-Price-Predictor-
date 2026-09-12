# Phase 26 — Deployment

**Target:** Streamlit Community Cloud (free).
**Pre-flight:** `python check_deployment.py` — 28 checks.

---

## The failure this phase is really about

A `.joblib` file is a **pickle** — a snapshot of live Python objects. It
stores the class paths and the internal attribute layout of the exact
scikit-learn version that created it.

Load it under a different scikit-learn and one of three things happens:

| | What you see |
|---|---|
| Best case | It works, with an `InconsistentVersionWarning` |
| Common | It loads, then throws a confusing `AttributeError` deep inside sklearn on the first `predict` |
| **Worst** | **It loads silently and predicts subtly different numbers** |

The third is the dangerous one. Nothing errors. The app is just wrong, and
nobody finds out.

### The fix is boring and complete: pin exact versions

```
pandas>=2.0          ← lets the cloud install pandas 3.1 next month
pandas==3.0.5        ← installs what your model was pickled with
```

Lower bounds are fine while you develop. They are a liability in production,
because the app you have not touched in six months breaks on its own when a
dependency releases a new version.

`check_deployment.py --write-requirements` writes the exact pins from your
current environment. **Run it on your own machine** — your library versions
are not the same as anyone else's.

### The check that catches it

`model_card.json` records the versions the model was saved with (Phase 19,
and this is what that was for). The script compares them to what is
installed:

```
4. VERSION MATCH  (the one that silently breaks predictions)
  PASS  python matches (3.11.15 saved, 3.11.15 here)
  PASS  sklearn matches (1.8.0 saved, 1.8.0 here)
  PASS  numpy matches (2.4.4 saved, 2.4.4 here)
  PASS  pandas matches (3.0.2 saved, 3.0.2 here)
```

If any of those FAIL, retrain before deploying. Do not deploy a model
pickled by a version you are no longer running.

---

## What else the pre-flight checks

Each one catches a problem that would otherwise cost a full build cycle —
push, wait three minutes, read a log, guess, push again.

| Check | The failure it prevents |
|---|---|
| Model is **committed**, not gitignored | The cloud only sees committed files. An app with no model file. The most common failure after version mismatch. |
| **Only one** dependency file | Community Cloud reads `uv.lock`, `Pipfile`, `environment.yml`, `requirements.txt`, `pyproject.toml` **in that order and uses only the first**. A leftover `Pipfile` silently overrides your `requirements.txt`. |
| No dev-only packages in `requirements.txt` | jupyterlab, xgboost and matplotlib are reinstalled on every cold start. Nothing breaks; the app is just slow to wake. |
| No data files committed | An 8.5 MB CSV in the repo, for nothing. |
| No secrets committed | `.env`, `*.key`, `secrets.toml`. |
| Model under 90 MB | GitHub refuses over 100 MB and warns over 25 MB. Ours is 1.6 MB — another dividend from Phase 18 choosing HistGB over Random Forest's 54.8 MB. |
| Working tree clean | Uncommitted work does not deploy. |
| The model actually loads and predicts | Catches everything above, together. |

---

## Deployment, step by step

### 1. Pin and commit

```bash
python check_deployment.py --write-requirements
python check_deployment.py          # expect 28 passed, 0 failed
git add -A
git commit -m "Phase 26: pin exact versions for deployment"
git push
```

### 2. Create the app

1. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
2. **Create app** → **Deploy a public app from GitHub**
3. Fill in:

| Field | Value |
|---|---|
| Repository | `<your-username>/House-Price-Predictor-` |
| Branch | `main` |
| **Main file path** | **`app/app.py`** |
| App URL | choose something short |

4. **Advanced settings → Python version → 3.11**

That last one matters. Community Cloud defaults to whatever Python it
currently prefers, and a pickle made on 3.11 is not guaranteed to load on
3.13. Set it to match `model_card.json`.

5. **Deploy.** First build takes 3–5 minutes.

### 3. Test the live app

Open the URL and try all four:

| Input | Expect |
|---|---|
| Andheri, 900 sqft, 2 BHK | ~₹2.5 crore, no warnings |
| Badlapur, 550 sqft, 1 BHK | ~₹22 lakh, no warnings |
| Malabar Hill, 4500 sqft, 4 BHK | ~₹51 crore, **4 warnings** |
| Any locality, area 100 | rejected with a readable message |

If the first three work and the fourth is rejected properly, the deployment
is correct end to end — model, preprocessing, validation and warnings.

### 4. Put the URL in the README

```markdown
**[Live app →](https://your-app.streamlit.app)**
```

Also add it to the GitHub repo's "Website" field, top right.

---

## Common deployment problems

**`ModuleNotFoundError: No module named 'predict'`**
The main file path is wrong, or `app.py`'s `sys.path` insert is not
resolving. Main file path must be `app/app.py`, not `app.py`.

**`FileNotFoundError: models/house_price_model.joblib`**
The model is gitignored. Check `git ls-files models/`. If it is empty, your
`.gitignore` is excluding it.

**`InconsistentVersionWarning` in the logs, then odd predictions**
Exactly the failure at the top of this document. Pin the versions and
redeploy.

**App boots, then dies with no message**
Out of memory. The free tier gives roughly 1 GB. If you had chosen Random
Forest in Phase 18 (162.8 MB in RAM) this would be a live risk; with HistGB
at 3.6 MB it is not.

**Build takes forever / times out**
`requirements.txt` has dev packages in it. The pre-flight check flags this.

**Changes do not appear**
Community Cloud rate-limits updates to **five per minute**. Wait, then use
"Reboot app" from the app menu.

**`ERROR: Could not find a version that satisfies the requirement`**
A pinned version does not exist for the cloud's Python version. Either
relax that one pin, or set the Python version to match your machine.

---

## Updating a deployed app

Community Cloud watches your GitHub branch. Push, and it redeploys.

```bash
# after retraining
python run_pipeline.py
pytest
python check_deployment.py --write-requirements   # versions may have moved
python check_deployment.py
git add -A && git commit -m "Retrain" && git push
```

**Always re-run `--write-requirements` after retraining.** If you upgraded
scikit-learn in between, the new pickle needs the new pin, and the old pin
would break it.

---

## Limits worth knowing

| | |
|---|---|
| Where apps run | United States, not configurable |
| Update rate limit | 5 per minute |
| Python | Only versions still receiving security updates |
| Dependency files | One only, first match in priority order wins |
| System packages | `packages.txt` in the repo root (we need none) |
| Inactivity | Unused apps sleep and wake on the next visit — the first load after that is slow |

---

## What this phase actually demonstrates

The deployment itself is a form and a button. What is worth explaining in an
interview is everything the pre-flight script checks:

- Understanding that a pickle is **version-bound**, and that the failure mode
  is silence rather than an error
- Recording versions **at save time** (Phase 19) so this check is possible at
  all
- Choosing a 1.6 MB model over a 54.8 MB one **because of where it has to
  run** (Phase 18)
- Separating runtime from development dependencies (Phase 23)
- Turning a five-minute build-and-guess loop into a two-second local check

Sources: [Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) ·
[Status and limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status) ·
[InconsistentVersionWarning](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.InconsistentVersionWarning.html)
