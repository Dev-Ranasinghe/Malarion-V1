"""
Test script to verify the image upload functionality
"""
import requests
from pathlib import Path

# Test with the sample images
sample_dir = Path('samples')

print("Testing Flask app image upload...")
print("=" * 50)

for img_file in sample_dir.glob('*.png'):
    print(f"\nTesting with: {img_file.name}")
    
    with open(img_file, 'rb') as f:
        files = {'pic': f}
        response = requests.post('http://127.0.0.1:5000/result', files=files)
    
    if response.status_code == 200:
        print(f"✓ Upload successful")
        # Extract prediction from response
        if 'Infected' in response.text:
            if 'Not Infected' in response.text:
                print("✓ Prediction: Not Infected")
            else:
                print("✓ Prediction: Infected")
    else:
        print(f"✗ Upload failed with status {response.status_code}")
        print(f"Error: {response.text[:200]}")

print("\n" + "=" * 50)
print("Test complete!")
