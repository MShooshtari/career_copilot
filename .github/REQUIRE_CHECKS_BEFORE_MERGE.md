# Require PR checks before merging

To block merging until all **pr-checks** jobs pass:

1. Open the repo on GitHub → **Settings** → **Branches**.
2. Under **Branch protection rules**, click **Add rule** (or edit the rule for `main` / your default branch).
3. Set **Branch name pattern** to `main` (or `master` if that’s your default).
4. Enable **Require status checks to pass before merging**.
5. Click **Add status checks** and add each job from the **pr-checks** workflow:
   - **Lint**
   - **Test (Python 3.11)**
   - **Test (Python 3.12)**
   - **Security (dependency audit)**
6. Save the rule.

After this, pull requests targeting that branch cannot be merged until all four checks are green.
