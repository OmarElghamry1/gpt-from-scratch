# Make gpt_from_scratch ready for GitHub

## Context
The folder isn't a git repo yet. It has editor/OS junk, no README, no .gitignore, a couple of small bugs in `gpt.py`, and the experiment results sitting in an `.rtf` file. Goal: a clean repo you can push.

## Changes

1. **`.gitignore`** (new): `.DS_Store`, `.idea/`, `.vscode/`, `.ipynb_checkpoints/`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`.

2. **`gpt.py` fixes**
   - Delete the stray unused imports `from re import X` and `from tarfile import BLOCKSIZE`.
   - Fix device detection: it currently reads `'mps' if torch.cuda.is_available() else 'cpu'`, so it never uses MPS on a Mac. Change to `'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'`, and use the same line in `Bigram.py`.

3. **`README.md`** (new, short):
   - What it is: a character-level GPT trained on Tiny Shakespeare, built step by step (bigram → self-attention → multi-head → feed-forward → residual blocks → LayerNorm → dropout), following Karpathy's "Let's build GPT".
   - Files: `Bigram.py`, `gpt.py`, `gpt_from_scratch.ipynb`, `Residual Network .ipynb` (uses `names.txt`), `input.txt`, `names.txt`.
   - How to run: `pip install torch` (plus `matplotlib` for the notebook), then `python gpt.py`.
   - Results: the loss table from `tracking_loss.rtf`, turned into a markdown table (stage → train/val loss at step 2700/4800).

4. **Remove `tracking_loss.rtf`** once its contents are in the README (or keep it; either way the README has the table).

5. **Keep the data files** (`input.txt` 1.1 MB, `names.txt` 224 KB). They're small and the scripts need them.

6. **Git init + first commit**: `git init`, `git add .`, commit "Initial commit: GPT from scratch" (no Claude mention). I won't add a remote or push; `gh` isn't installed. After you create the empty repo on GitHub, you run `git remote add origin <url> && git push -u origin main`.

Note: your global git email is `omarm.elghamry@gmail.com`. Make sure that's the one on your GitHub account.

## Verification
- `git status` is clean and `git ls-files` shows no `.DS_Store`, `.idea`, `.vscode` or checkpoints.
- `python -c "import ast; ast.parse(open('gpt.py').read())"` passes. Optionally, run `python gpt.py` with a small `max_iters` to check it starts.
