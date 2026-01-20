# GitHub Repository Setup Instructions

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name your repository (e.g., `crypto-trading-backtest`)
5. Choose public or private
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 2: Push Your Code to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME and YOUR_REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Rename branch to main (optional, GitHub uses 'main' by default)
git branch -M main

# Push your code
git push -u origin main
```

## Alternative: Using SSH

If you prefer SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Files Included

- `app.py` - FastAPI web service application
- `backtest_engine.py` - Core backtest logic
- `static/index.html` - Web interface
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules

## Files Excluded (.gitignore)

- `__pycache__/` - Python cache files
- `*.png` - Generated plot images
- Virtual environment folders
- IDE configuration files
- `BinanceUSA_bot.ipynb` - Original notebook (not needed for web service)

## Next Steps After Pushing

1. Update the README.md with your actual GitHub repo URL
2. Consider adding a license file (MIT, Apache, etc.)
3. Add topics/tags to your GitHub repo for discoverability
4. Consider adding GitHub Actions for CI/CD if needed
