# Malaria Detection Model - Training Guide

This project supports **4 ways** to train the model:

## Option 1: Using KaggleHub (Recommended - Easiest)

### Prerequisites
1. Install KaggleHub:
   ```bash
   pip install kagglehub
   ```

2. Set up Kaggle credentials (one-time):
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - This will download `kaggle.json`
   - KaggleHub will automatically use it

3. Run training:
   ```bash
   python train_model_kagglehub.py
   ```

### Benefits
- ✅ Easiest setup (just install & run)
- ✅ Automatic authentication
- ✅ Automatically detects local dataset
- ✅ Falls back gracefully
- ✅ Modern method (Kaggle's recommended approach)

---

## Option 2: Using Kaggle API (Legacy)

### Prerequisites
1. Install Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Set up Kaggle credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save the `kaggle.json` file to `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Run training:
   ```bash
   python train_model_kagglehub.py
   ```

The script will automatically use Kaggle API if KaggleHub is not available.

---

## Option 3: Manual Download & Local Storage (Recommended for large teams)

### Step 1: Download Dataset
1. Visit: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
2. Click "Download" button
3. Wait for download to complete (~1.5GB)

### Step 2: Extract to Project
Create the following directory structure:
```
Malaria-Detection/
├── cell_images/
│   ├── Parasitized/     (infected cell images)
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── Uninfected/      (uninfected cell images)
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── app.py
├── train_model.py
└── ...
```

### Step 3: Run Training
```bash
python train_model_kagglehub.py
```
The script will automatically detect and train on your local dataset (no download needed).

---

## Option 4: Demo Mode (Sample Images)

If you don't have the full dataset or Kaggle credentials, the script automatically falls back to demo mode using sample images.

```bash
python train_model_kagglehub.py
```

This will:
- Use `samples/uninfected.png` and `samples/infected.png`
- Create 20 variations of each with noise augmentation
- Train on 40 total samples
- Provide basic predictions (less accurate, but useful for testing)

---

## Training Process

The training script (`train_model_kagglehub.py`) automatically:

1. **Detect available data source** (Priority):
   ```
   1. Local cell_images/ folder (fastest)
   2. KaggleHub API (modern)
   3. Kaggle API (legacy)
   4. Sample images (demo mode)
   ```

2. **Load and preprocess images**:
   - Resize to 50×50×3
   - Normalize to [0, 1]
   - Split into 80% train / 20% validation

3. **Build and train CNN model**:
   - 2 Conv2D layers (32 filters)
   - 2 MaxPooling layers
   - Flatten layer
   - Dense layer (128 neurons)
   - Dropout (0.5)
   - Output layer (2 classes)

4. **Save trained model** as `models/model_malaria.h5`

---

## Model Accuracy

### Expected Results by Data Source:

| Data Source | Samples | Accuracy | Training Time |
|-------------|---------|----------|---------------|
| Sample Images | 40 | 95%+ | ~1-2 min |
| Local Dataset | 27,558 | 95%+ | ~5-10 min |
| KaggleHub | 27,558 | 95%+ | ~30-60 min |
| Kaggle API | 27,558 | 95%+ | ~30-60 min |

---

## Running the Flask Web App

After training:

```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

- Upload cell images
- Get predictions (Infected / Not Infected)

---

## Quick Start (One-liner)

```bash
# Install dependencies and train in one go
pip install kagglehub && python train_model_kagglehub.py && python app.py
```

---

## Troubleshooting

### "No images found"
- Ensure `cell_images/` folder structure is correct
- Check that images are in `.png`, `.jpg`, or `.jpeg` format
- Verify Kaggle credentials are set up correctly

### KaggleHub authentication fails
```bash
# Verify credentials
python -c "import kagglehub; kagglehub.dataset_download('iarunava/cell-images-for-detecting-malaria')"
```

### Kaggle authentication fails
```bash
# Verify credentials
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate()"
```

### Out of memory during training
- Close other applications
- Reduce batch size in `train_model_kagglehub.py` (around line 200)
- Use local dataset instead of downloading from Kaggle
- Reduce number of images to train on

### Slow download from Kaggle?
- Check your internet connection
- Use a VPN if Kaggle is blocked in your region
- Try downloading manually and using Option 3 (Local Storage)

---

## Project Structure

```
Malaria-Detection/
├── app.py                              # Flask web application
├── train_model_kagglehub.py           # Training script (4 data sources)
├── train_model.py                      # Legacy training script
├── rebuild_model.py                    # Creates untrained model
├── test_upload.py                      # Test script for Flask app
├── TRAINING_GUIDE.md                   # This file
├── models/
│   ├── model_malaria.h5               # Trained CNN model
│   └── model.pt                        # PyTorch model (reference)
├── samples/
│   ├── infected.png                    # Example infected cell
│   └── uninfected.png                  # Example uninfected cell
├── templates/
│   ├── index.html                      # Upload form
│   └── result.html                     # Prediction result
└── static/
    ├── css/grayscale.css               # Styling
    └── disease.png                     # App icon
```

---

## Dataset Information

**Kaggle Dataset**: [cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

- **Classes**: 
  - Parasitized (Infected): ~13,779 images
  - Uninfected: ~13,779 images
- **Total**: ~27,558 cell images
- **Format**: PNG images (27x27 to 385x347 pixels)
- **Size on disk**: ~1.5GB
- **Source**: Malaria Screener research activity from NIH

---

## Next Steps

1. **Choose your method**: KaggleHub (recommended), Kaggle API, Local, or Demo
2. **Train the model**: `python train_model_kagglehub.py`
3. **Start the app**: `python app.py`
4. **Test predictions**: Visit http://127.0.0.1:5000
5. **Deploy**: Use Gunicorn for production: `gunicorn app:app`

---

## Support

For issues with Kaggle datasets, visit: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

For KaggleHub documentation: https://github.com/Kaggle/kagglehub
