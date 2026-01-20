"""
Train the Malaria Detection model with KaggleHub support.
Supports three data sources (in priority order):
1. KaggleHub API - Modern, easier setup
2. Local storage - cell_images/ folder
3. Sample images - Demo mode with 2 sample images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import os
from pathlib import Path
from skimage import transform
from sklearn.model_selection import train_test_split
import sys

# Build the model architecture
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(50, 50, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def load_images_from_directory(directory, label, resize_shape=(50, 50, 3), max_images=None):
    """Load all images from a directory"""
    X = []
    y = []
    
    img_dir = Path(directory)
    if not img_dir.exists():
        return X, y
    
    image_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg'))
    
    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            img_array = transform.resize(img_array, resize_shape, anti_aliasing=True)
            img_array /= 255
            X.append(img_array)
            y.append(label)
            
            if idx % 100 == 0:
                print(f"    Loaded {idx} images...", end='\r')
        except Exception as e:
            print(f"  ⚠ Skipped {img_path.name}: {str(e)}")
    
    return X, y

def download_kagglehub_dataset():
    """Download Kaggle dataset using KaggleHub API (modern method)"""
    try:
        import kagglehub
        
        print("\n📥 Downloading dataset from Kaggle using KaggleHub...")
        print("  Note: This may take a few minutes (~1.5GB download)")
        
        # Download the malaria dataset
        dataset_path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
        print(f"  ✓ Dataset downloaded to: {dataset_path}")
        
        # Navigate to cell_images folder
        cell_images_path = Path(dataset_path) / "cell_images"
        if cell_images_path.exists():
            return str(cell_images_path)
        else:
            # Try alternate paths
            for potential_path in [
                Path(dataset_path),
                Path(dataset_path).parent / "cell_images",
                Path(dataset_path) / "cell-images-for-detecting-malaria"
            ]:
                if potential_path.exists():
                    return str(potential_path)
        
        return dataset_path
    except ImportError:
        print("  ⚠ KaggleHub not installed.")
        print("    Install with: pip install kagglehub")
        return None
    except Exception as e:
        print(f"  ⚠ Failed to download with KaggleHub: {str(e)}")
        return None

def download_kaggle_api_dataset():
    """Download Kaggle dataset using old Kaggle API (fallback)"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("\n📥 Downloading dataset from Kaggle using legacy API...")
        print("  Note: This may take a few minutes (~1.5GB download)")
        
        api = KaggleApi()
        api.authenticate()
        
        # Download the malaria dataset
        api.dataset_download_files('iarunava/cell-images-for-detecting-malaria', path='.', unzip=True)
        print("  ✓ Dataset downloaded and extracted!")
        return "cell_images"
    except ImportError:
        print("  ⚠ Kaggle API not installed. Install with: pip install kaggle")
        return None
    except Exception as e:
        print(f"  ⚠ Failed to download with Kaggle API: {str(e)}")
        return None

print("=" * 70)
print("MALARIA DETECTION MODEL TRAINING")
print("=" * 70)

X_train = []
y_train = []
training_source = None

# Priority 1: Check for local cell_images directory
print("\n1️⃣  Checking for local dataset (cell_images/)...")
cell_images_path = Path("cell_images")
uninfected_dir = cell_images_path / "Uninfected"
infected_dir = cell_images_path / "Parasitized"

if uninfected_dir.exists() and infected_dir.exists():
    print("  ✓ Local dataset found!")
    training_source = "local"
    
    # Load uninfected images
    print("  Loading uninfected images...")
    X_uninfected, y_uninfected = load_images_from_directory(uninfected_dir, 0)
    X_train.extend(X_uninfected)
    y_train.extend(y_uninfected)
    print(f"  ✓ Loaded {len(X_uninfected)} uninfected images")
    
    # Load infected images
    print("  Loading infected/parasitized images...")
    X_infected, y_infected = load_images_from_directory(infected_dir, 1)
    X_train.extend(X_infected)
    y_train.extend(y_infected)
    print(f"  ✓ Loaded {len(X_infected)} infected images")
    
    print(f"  📊 Total images: {len(X_train)}")
else:
    print("  ✗ Local dataset not found")
    
    # Priority 2: Try KaggleHub (modern method)
    print("\n2️⃣  Attempting to download from Kaggle using KaggleHub...")
    dataset_path = download_kagglehub_dataset()
    
    if dataset_path:
        training_source = "kagglehub"
        cell_images_path = Path(dataset_path)
        uninfected_dir = cell_images_path / "Uninfected"
        infected_dir = cell_images_path / "Parasitized"
        
        # Load uninfected images
        print("  Loading uninfected images...")
        X_uninfected, y_uninfected = load_images_from_directory(uninfected_dir, 0)
        X_train.extend(X_uninfected)
        y_train.extend(y_uninfected)
        print(f"  ✓ Loaded {len(X_uninfected)} uninfected images")
        
        # Load infected images
        print("  Loading infected/parasitized images...")
        X_infected, y_infected = load_images_from_directory(infected_dir, 1)
        X_train.extend(X_infected)
        y_train.extend(y_infected)
        print(f"  ✓ Loaded {len(X_infected)} infected images")
        
        print(f"  📊 Total images: {len(X_train)}")
    else:
        print("  ✗ KaggleHub download failed")
        
        # Priority 3: Try legacy Kaggle API (fallback)
        print("\n3️⃣  Attempting to download from Kaggle using legacy API...")
        dataset_path = download_kaggle_api_dataset()
        
        if dataset_path:
            training_source = "kaggle_api"
            cell_images_path = Path(dataset_path)
            uninfected_dir = cell_images_path / "Uninfected"
            infected_dir = cell_images_path / "Parasitized"
            
            # Load uninfected images
            print("  Loading uninfected images...")
            X_uninfected, y_uninfected = load_images_from_directory(uninfected_dir, 0)
            X_train.extend(X_uninfected)
            y_train.extend(y_uninfected)
            print(f"  ✓ Loaded {len(X_uninfected)} uninfected images")
            
            # Load infected images
            print("  Loading infected/parasitized images...")
            X_infected, y_infected = load_images_from_directory(infected_dir, 1)
            X_train.extend(X_infected)
            y_train.extend(y_infected)
            print(f"  ✓ Loaded {len(X_infected)} infected images")
            
            print(f"  📊 Total images: {len(X_train)}")
        else:
            print("  ✗ Kaggle API download failed")
            
            # Priority 4: Fall back to sample images
            print("\n4️⃣  Falling back to sample images (demo mode)...")
            training_source = "samples"
            
            samples_dir = "samples"
            
            # Load uninfected sample
            uninfected_path = os.path.join(samples_dir, "uninfected.png")
            if os.path.exists(uninfected_path):
                img = Image.open(uninfected_path).convert('RGB')
                img_array = np.array(img, dtype=np.float32)
                img_array = transform.resize(img_array, (50, 50, 3), anti_aliasing=True)
                img_array /= 255
                
                # Create variations
                for i in range(20):
                    noisy_img = img_array + np.random.normal(0, 0.02, img_array.shape)
                    noisy_img = np.clip(noisy_img, 0, 1)
                    X_train.append(noisy_img)
                    y_train.append(0)
                print(f"  ✓ Loaded uninfected.png - created 20 variations")
            
            # Load infected sample
            infected_path = os.path.join(samples_dir, "infected.png")
            if os.path.exists(infected_path):
                img = Image.open(infected_path).convert('RGB')
                img_array = np.array(img, dtype=np.float32)
                img_array = transform.resize(img_array, (50, 50, 3), anti_aliasing=True)
                img_array /= 255
                
                # Create variations
                for i in range(20):
                    noisy_img = img_array + np.random.normal(0, 0.02, img_array.shape)
                    noisy_img = np.clip(noisy_img, 0, 1)
                    X_train.append(noisy_img)
                    y_train.append(1)
                print(f"  ✓ Loaded infected.png - created 20 variations")
            
            print(f"  📊 Total samples: {len(X_train)}")

if not X_train:
    print("\n✗ ERROR: No images found to train on!")
    print("\nTo fix, install one of the following:")
    print("  - KaggleHub: pip install kagglehub")
    print("  - Kaggle API: pip install kaggle")
    print("  OR manually download and extract to cell_images/")
    sys.exit(1)

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"\n📊 Dataset summary:")
print(f"   Total images: {X_train.shape[0]}")
print(f"   Image shape: {X_train.shape[1:]}")
print(f"   Source: {training_source}")

# One-hot encode labels
y_train_encoded = keras.utils.to_categorical(y_train, 2)

# Split into train and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42
)

print(f"   Training set: {X_train_split.shape[0]} images")
print(f"   Validation set: {X_val.shape[0]} images")

# Build and train model
print("\n" + "=" * 70)
print("BUILDING AND TRAINING MODEL")
print("=" * 70 + "\n")

model = build_model()
print(model.summary())

# Determine epochs based on dataset size
if training_source == "samples":
    epochs = 50
else:
    epochs = 20

print(f"\nTraining for {epochs} epochs...")

history = model.fit(
    X_train_split,
    y_train_split,
    epochs=epochs,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate on validation set
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70 + "\n")

model.save("models/model_malaria.h5")
print("✓ Model saved as models/model_malaria.h5")

# Test predictions on sample images
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70 + "\n")

sample_uninfected = "samples/uninfected.png"
sample_infected = "samples/infected.png"

if os.path.exists(sample_uninfected):
    img = Image.open(sample_uninfected).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_array = transform.resize(img_array, (50, 50, 3), anti_aliasing=True) / 255
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)
    class_name = "UNINFECTED" if pred[0][0] > pred[0][1] else "INFECTED"
    confidence = max(pred[0]) * 100
    print(f"Uninfected sample: {class_name} ({confidence:.2f}% confidence)")

if os.path.exists(sample_infected):
    img = Image.open(sample_infected).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_array = transform.resize(img_array, (50, 50, 3), anti_aliasing=True) / 255
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)
    class_name = "UNINFECTED" if pred[0][0] > pred[0][1] else "INFECTED"
    confidence = max(pred[0]) * 100
    print(f"Infected sample: {class_name} ({confidence:.2f}% confidence)")

print("\n" + "=" * 70)
print("✓ TRAINING COMPLETE!")
print("=" * 70)

print(f"\n📝 Training Source: {training_source.upper()}")
if training_source == "samples":
    print("\nNote: Model trained on sample images only (demo mode)")
    print("\nFor better accuracy, use the full Kaggle dataset:")
    print("\n  Option 1 - KaggleHub (Recommended):")
    print("    pip install kagglehub")
    print("    python train_model.py")
    print("\n  Option 2 - Kaggle API:")
    print("    pip install kaggle")
    print("    python train_model.py")
    print("\n  Option 3 - Manual download:")
    print("    Download from: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
    print("    Extract to: cell_images/")
