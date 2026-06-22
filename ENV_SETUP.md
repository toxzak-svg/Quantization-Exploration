# Local Environment Setup for Colab Pipeline

## Quick Start

### 1. Create `.env` file in your project root

Copy `.env.example` to `.env` and fill in your actual tokens:

```bash
cp .env.example .env
# Then edit .env with your actual credentials
```

### 2. Get your tokens

- **Hugging Face Token**: Visit https://huggingface.co/settings/tokens → New token → Copy
- **GitHub Token** (optional, only for auto-push): Visit https://github.com/settings/tokens → Generate new token → Copy

### 3. Set the `.env` file

```bash
# .env (do not commit this to git!)
HF_TOKEN=hf_abc123xyz...
GH_TOKEN=ghp_xyz123abc...
HF_REPO_ID=your-username/your-repo
```

### 4. Add `.env` to `.gitignore`

```bash
echo ".env" >> .gitignore
```

### 5. Run the notebook

The notebook will automatically detect the local environment and read secrets from `.env`.

---

## How it works

### Secret Priority (in order)
1. **Environment variables** (e.g., `export HF_TOKEN=...`)
   - Useful for CI/CD pipelines and Colab
2. **`.env` file** (local only, in project root)
   - Useful for local development
3. **Interactive prompt** via `getpass()` (secure, not stored)
   - Fallback if token not found above

### Local vs Colab Detection

```python
IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    # Colab: mount Drive, save to /content/drive/MyDrive/...
else:
    # Local: save to ./artifacts
```

---

## Security Notes

⚠️ **Never commit `.env` to version control!**
- Add `.env` to `.gitignore` immediately
- Use SSH keys for GitHub instead of tokens when possible
- Tokens in CI/CD should use secrets management (GitHub Secrets, GitLab CI Variables, etc.)
- For Colab, environment variables are session-only and cleared on disconnect

---

## Troubleshooting

**"HF_TOKEN not set"**
- Check `.env` exists in project root
- Check file contains `HF_TOKEN=hf_...` (no spaces around `=`)
- Ensure `.env` is readable (`chmod 600 .env` on Unix)

**"HF_REPO_ID invalid"**
- Format must be `username/repo-name` (exactly)
- Ensure the repo exists on Hugging Face Hub
- Ensure your token has write access to that repo

**"git commit failed"**
- Ensure `git` is installed (`pip install gitpython`)
- Check that files to commit exist and are readable

---

## Example Workflow

```bash
# 1. Clone / set up the project
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Create .env from template
cp .env.example .env
# Edit .env with your tokens
nano .env  # or use your editor

# 3. Run the notebook (local Jupyter)
jupyter notebook notebook/colab_hf_github_pipeline.ipynb

# 4. Or use Colab (upload notebook, set env vars manually)
# In Colab cell 1: import os; os.environ['HF_TOKEN'] = 'hf_...'
```

---

## For Colab Users

If using the notebook in Colab, set environment variables in the first cell:

```python
import os
os.environ['HF_TOKEN'] = 'hf_your_token'
os.environ['GH_TOKEN'] = 'ghp_your_github_token'
os.environ['HF_REPO_ID'] = 'your-username/your-repo'
```

Or use the secure prompt (notebook will ask you if env var not set).
