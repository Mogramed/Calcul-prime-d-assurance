# Artifacts Policy

This directory stores runtime outputs only:

1. model bundles (`models/`)
2. data science exports (`ds/`)
3. submissions and experiment results

Repository policy is **code-only**. Generated artifact files are ignored by
default via `.gitignore`.

If you need to keep a specific artifact for documentation, commit it explicitly
with a clear reason in the PR description.
