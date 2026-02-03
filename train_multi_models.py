"""
Train multiple malaria detection models with different architectures.
Supports: Simple CNN, VGG16, ResNet50, DenseNet121, MobileNetV2
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
from model_loader import (
    build_transfer_learning_model,
    build_simple_cnn_model,
    save_model,
    save_model_accuracy,
    MODEL_ARCHITECTURES,
    MODEL_NAMES_DISPLAY
)

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

def load_dataset():
    """Load the dataset with multiple priority sources"""
    print("\n" + "=" * 70)
    print("LOADING DATASET")
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
        return None, None, None
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # One-hot encode labels
    y_train_encoded = keras.utils.to_categorical(y_train, 2)
    
    # Split into train and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_encoded, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Dataset summary:")
    print(f"   Total images: {X_train.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]}")
    print(f"   Source: {training_source}")
    print(f"   Training set: {X_train_split.shape[0]} images")
    print(f"   Validation set: {X_val.shape[0]} images")
    
    return (X_train_split, X_val, y_train_split, y_val), training_source

def train_model(model_name, X_train, X_val, y_train, y_val, epochs=20):
    """Train a single model"""
    print("\n" + "=" * 70)
    print(f"TRAINING {MODEL_NAMES_DISPLAY.get(model_name, model_name).upper()}")
    print("=" * 70 + "\n")
    
    # Build model
    if model_name == 'simple':
        model = build_simple_cnn_model()
        print(f"Building Simple CNN model...")
    else:
        model = build_transfer_learning_model(model_name)
        print(f"Building {MODEL_NAMES_DISPLAY.get(model_name, model_name)} with transfer learning...")
    
    print(model.summary())
    
    print(f"\nTraining for {epochs} epochs...")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Validation Loss: {val_loss:.4f}")
    print(f"✓ Validation Accuracy: {val_accuracy*100:.2f}%")
    
    # Save model
    save_model(model, model_name)
    
    # Save model accuracy to metadata
    accuracy_percent = val_accuracy * 100
    save_model_accuracy(model_name, accuracy_percent)
    
    return model, accuracy_percent

def main():
    print("\n" + "=" * 70)
    print("MALARIA DETECTION MODEL TRAINING")
    print("Multiple Architecture Support")
    print("=" * 70)
    
    # Load dataset
    dataset_info = load_dataset()
    if dataset_info[0] is None:
        sys.exit(1)
    
    X_train_split, X_val, y_train_split, y_val = dataset_info[0]
    training_source = dataset_info[1]
    
    # Determine epochs based on dataset size
    if training_source == "samples":
        epochs = 50
    else:
        epochs = 20
    
    # Train all models
    models_to_train = ['simple', 'vgg16', 'resnet50', 'densenet121', 'mobilenetv2']
    results = {}
    
    print("\n" + "=" * 70)
    print(f"TRAINING {len(models_to_train)} MODELS")
    print("=" * 70)
    
    for model_name in models_to_train:
        try:
            print(f"\n\n{'='*70}")
            print(f"Training model: {MODEL_NAMES_DISPLAY.get(model_name, model_name)}")
            print(f"{'='*70}")
            
            model, accuracy = train_model(
                model_name, X_train_split, X_val, y_train_split, y_val, epochs=epochs
            )
            results[model_name] = accuracy
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70 + "\n")
    
    print("Model Performance:")
    for model_name in models_to_train:
        if results[model_name] is not None:
            acc = results[model_name]
            print(f"  ✓ {MODEL_NAMES_DISPLAY.get(model_name, model_name):20} - Accuracy: {acc:.2f}%")
        else:
            print(f"  ✗ {MODEL_NAMES_DISPLAY.get(model_name, model_name):20} - Failed")
    
    print(f"\n📝 Training Source: {training_source.upper()}")
    if training_source == "samples":
        print("\nNote: Models trained on sample images only (demo mode)")
        print("\nFor better accuracy, use the full Kaggle dataset:")
        print("\n  Option 1 - KaggleHub (Recommended):")
        print("    pip install kagglehub")
        print("    python train_multi_models.py")
        print("\n  Option 2 - Kaggle API:")
        print("    pip install kaggle")
        print("    python train_multi_models.py")
        print("\n  Option 3 - Manual download:")
        print("    Download from: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
        print("    Extract to: cell_images/")
    
    print("\n" + "=" * 70)
    print("✓ ALL MODELS TRAINED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
