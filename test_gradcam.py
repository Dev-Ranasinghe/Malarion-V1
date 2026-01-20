#!/usr/bin/env python3
"""Test Grad-CAM generation independently"""

import sys
sys.path.insert(0, '/Users/dev/Desktop/MALARIAN V1/Malaria-Detection')

from app import loaded_model, generate_grad_cam, create_heatmap_overlay, array_to_base64
import numpy as np

print("\n" + "="*60)
print("TESTING GRAD-CAM FUNCTIONALITY")
print("="*60 + "\n")

# Create a random test image
test_img = np.random.rand(1, 50, 50, 3).astype(np.float32)
print(f"✓ Test image created: {test_img.shape}")

# Test model prediction
print("\n[1/4] Testing model prediction...")
try:
    pred = loaded_model.predict(test_img, verbose=0)
    pred_class = int(np.argmax(pred))
    print(f"✓ Model prediction successful: class {pred_class}, confidence: {np.max(pred)*100:.1f}%")
except Exception as e:
    print(f"✗ Model prediction failed: {e}")
    sys.exit(1)

# Test Grad-CAM
print("\n[2/4] Testing Grad-CAM generation...")
try:
    cam = generate_grad_cam(loaded_model, test_img, pred_class)
    if cam is not None:
        print(f"✓ Grad-CAM generated: shape {cam.shape}, min: {cam.min():.4f}, max: {cam.max():.4f}")
    else:
        print("✗ Grad-CAM is None!")
        sys.exit(1)
except Exception as e:
    print(f"✗ Grad-CAM generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test heatmap creation
print("\n[3/4] Testing heatmap overlay creation...")
try:
    overlay, heatmap, cam_resized = create_heatmap_overlay(None, cam, test_img)
    print(f"✓ Heatmap overlay created:")
    print(f"  - Overlay shape: {overlay.shape}")
    print(f"  - Heatmap shape: {heatmap.shape}")
    print(f"  - CAM resized shape: {cam_resized.shape}")
except Exception as e:
    print(f"✗ Heatmap creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test base64 encoding
print("\n[4/4] Testing base64 encoding...")
try:
    b64_overlay = array_to_base64(overlay)
    b64_heatmap = array_to_base64(heatmap)
    print(f"✓ Base64 encoding successful:")
    print(f"  - Overlay base64 length: {len(b64_overlay)}")
    print(f"  - Heatmap base64 length: {len(b64_heatmap)}")
except Exception as e:
    print(f"✗ Base64 encoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print("="*60 + "\n")
print("The Grad-CAM visualization is working correctly!")
print("Try uploading an image to http://127.0.0.1:5000\n")
