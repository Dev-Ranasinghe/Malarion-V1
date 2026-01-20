# KaggleHub Setup Guide

## What is KaggleHub?

KaggleHub is Kaggle's official Python library for downloading datasets and code. It's the modern replacement for the old Kaggle API.

**Advantages**:
- ✅ Easier setup (no manual file management)
- ✅ Official library (better maintained)
- ✅ Automatic credential detection
- ✅ Better error messages
- ✅ Supports both datasets and competition data

---

## Installation

### Step 1: Install KaggleHub

```bash
pip install kagglehub
```

### Step 2: Set Up Kaggle Credentials (One-time)

You have **two options**:

#### Option A: Create Kaggle API Token (Recommended)

1. Go to https://www.kaggle.com/account
2. Scroll down to "API" section
3. Click "Create New API Token"
4. This will download a `kaggle.json` file
5. Move it to the correct location:

**On macOS/Linux**:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows**:
```bash
# In PowerShell or Command Prompt
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

#### Option B: Use Environment Variables

Instead of the file, you can set environment variables:

**On macOS/Linux**:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

**On Windows (PowerShell)**:
```powershell
[Environment]::SetEnvironmentVariable("KAGGLE_USERNAME", "your_username", "User")
[Environment]::SetEnvironmentVariable("KAGGLE_KEY", "your_api_key", "User")
```

---

## Verify Installation

Test that KaggleHub is working:

```bash
python -c "import kagglehub; print('✓ KaggleHub installed successfully')"
```

Test that credentials are set up:

```bash
python -c "import kagglehub; print(kagglehub.dataset_download('iarunava/cell-images-for-detecting-malaria'))"
```

If successful, you'll see a path like: `/Users/yourname/.cache/kagglehub/datasets/iarunava/cell-images-for-detecting-malaria/versions/1`

---

## Training with KaggleHub

Once installed and configured, simply run:

```bash
python train_model_kagglehub.py
```

The script will:
1. Check for local `cell_images/` folder
2. Try to download from KaggleHub if not found
3. Fall back to legacy Kaggle API if needed
4. Use sample images if all else fails

---

## Troubleshooting

### Error: "Invalid API key"

This means your credentials are not set up correctly.

```bash
# Check if kaggle.json exists
cat ~/.kaggle/kaggle.json

# Or check environment variables
echo $KAGGLE_USERNAME
echo $KAGGLE_KEY
```

### Error: "Dataset not found"

Make sure you're using the correct dataset name:
- Correct: `iarunava/cell-images-for-detecting-malaria`
- Incorrect: `ialopez/sarscov2ct` (different dataset)

### Slow download?

- Check your internet connection
- KaggleHub caches downloads, so subsequent runs are faster
- Downloaded datasets are stored in: `~/.cache/kagglehub/`

### How to clear cached datasets?

```bash
# Remove specific dataset
rm -rf ~/.cache/kagglehub/datasets/iarunava/cell-images-for-detecting-malaria/

# Remove all cached data
rm -rf ~/.cache/kagglehub/
```

---

## Using Downloaded Data Locally

After the first download, you can reuse the data:

1. Find where it was downloaded:
   ```bash
   python -c "import kagglehub; print(kagglehub.dataset_download('iarunava/cell-images-for-detecting-malaria'))"
   ```

2. Copy to your project:
   ```bash
   cp -r ~/.cache/kagglehub/datasets/iarunava/cell-images-for-detecting-malaria/versions/1 ./cell_images
   ```

3. Next time, the script will use the local copy (no download needed)

---

## Common Commands

```bash
# List available datasets
kagglehub dataset list

# Download a specific dataset
kagglehub dataset download "iarunava/cell-images-for-detecting-malaria"

# Download and get path
python -c "import kagglehub; path = kagglehub.dataset_download('iarunava/cell-images-for-detecting-malaria'); print(f'Downloaded to: {path}')"
```

---

## More Information

- **KaggleHub GitHub**: https://github.com/Kaggle/kagglehub
- **Official Documentation**: https://github.com/Kaggle/kagglehub/blob/main/README.md
- **Malaria Dataset**: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

---

## Next Steps

1. Install KaggleHub: `pip install kagglehub`
2. Set up credentials (follow Option A or B above)
3. Verify: `python -c "import kagglehub; print('Ready!')"`
4. Train: `python train_model_kagglehub.py`
5. Run app: `python app.py`
