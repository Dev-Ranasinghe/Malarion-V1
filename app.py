from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import re
import os
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image
from skimage import transform
import cv2
import base64
from model_loader import (
    load_model as load_custom_model, 
    get_available_models, 
    MODEL_NAMES_DISPLAY,
    get_model_accuracy,
    save_model_accuracy
)

# Get the absolute path to the app directory
app_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(app_dir, 'templates')
static_dir = os.path.join(app_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for static files
app.jinja_env.auto_reload = True  # Enable template auto-reload

# Store the currently loaded model and its name
current_model = None
current_model_name = None

def load_model_by_name(model_name):
    """Load a model by name and cache it."""
    global current_model, current_model_name
    
    try:
        if model_name == 'simple':
            current_model = load_model("models/model_malaria.h5")
        else:
            current_model = load_custom_model(model_name)
        current_model_name = model_name
        return True
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return False

def ValuePredictor(np_arr, model_name='simple'):   
    """Make prediction using the specified model."""
    if current_model is None or current_model_name != model_name:
        if not load_model_by_name(model_name):
            # Fallback to simple model
            load_model_by_name('simple')
    
    result = current_model.predict(np_arr, verbose=0)
    return result[0]

def image_preprocess(img_file):
  new_shape = (50, 50, 3)
  # Read the uploaded file from Flask FileStorage
  img = Image.open(img_file.stream).convert('RGB')
  # Convert PIL Image to numpy array
  image_array = np.array(img, dtype=np.float32)
  # Resize the image
  image_array = transform.resize(image_array, new_shape, anti_aliasing=True)
  # Normalize to [0, 1]
  image_array /= 255
  # Add batch dimension
  image_array = np.expand_dims(image_array, axis=0)
  return image_array

def generate_grad_cam(model, img_array, pred_index=None):
    """Generate Grad-CAM visualization for model explainability"""
    try:
        print("[DEBUG] Finding conv layer...")
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                print(f"[DEBUG] Found conv layer: {layer.name}")
                break
        
        if last_conv_layer is None:
            print("[DEBUG] No conv layer found!")
            return None
        
        print(f"[DEBUG] Image array shape: {img_array.shape}")
        
        # Convert image array to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        if len(img_tensor.shape) == 3:
            img_tensor = tf.expand_dims(img_tensor, 0)
        
        print("[DEBUG] Creating functional grad model...")
        # Build a simple functional model from scratch that uses the original layers
        input_layer = tf.keras.Input(shape=(50, 50, 3))
        x = input_layer
        conv_output = None
        
        # Pass input through all layers, capturing conv output
        for layer in model.layers:
            x = layer(x)
            if layer == last_conv_layer:
                conv_output = x
        
        # Create the model with conv output and final predictions
        grad_model = tf.keras.Model(inputs=input_layer, outputs=[conv_output, x])
        print("[DEBUG] Grad model created successfully")
        
        print("[DEBUG] Computing gradients with GradientTape...")
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            print(f"[DEBUG] Predictions shape: {predictions.shape}, values: {predictions}")
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Handle both tensor and int pred_index
            pred_idx = int(pred_index.numpy() if hasattr(pred_index, 'numpy') else pred_index)
            print(f"[DEBUG] Using class index: {pred_idx}")
            class_channel = predictions[:, pred_idx]
            print(f"[DEBUG] Class channel: {class_channel}")
        
        print("[DEBUG] Computing gradients...")
        grads = tape.gradient(class_channel, conv_outputs)
        print(f"[DEBUG] Gradients computed: {grads is not None}")
        
        if grads is None:
            print("[DEBUG] No gradients - trying alternative approach...")
            return None
        
        print(f"[DEBUG] Gradients shape: {grads.shape}")
        print(f"[DEBUG] Conv outputs shape: {conv_outputs.shape}")
        
        # Compute weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(f"[DEBUG] Pooled grads shape: {pooled_grads.shape}")
        
        # Get activation map
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()
        
        print(f"[DEBUG] Conv outputs shape (np): {conv_outputs_np.shape}")
        
        # Compute CAM manually
        cam = np.zeros((conv_outputs_np.shape[0], conv_outputs_np.shape[1]), dtype=np.float32)
        for i in range(conv_outputs_np.shape[2]):
            cam += pooled_grads_np[i] * conv_outputs_np[:, :, i]
        
        # Normalize
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
            
        print(f"[DEBUG] Final CAM shape: {cam.shape}, min: {cam.min()}, max: {cam.max()}")
        return cam
        
    except Exception as e:
        print(f"[DEBUG] ERROR in generate_grad_cam: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_heatmap_overlay(original_img, cam, img_array):
    """Create a heatmap overlay on the original image"""
    try:
        print("[DEBUG] Creating heatmap overlay...")
        
        # Denormalize the image
        img_display = (img_array[0] * 255).astype(np.uint8)
        print(f"[DEBUG] Image display shape: {img_display.shape}, dtype: {img_display.dtype}")
        
        # Convert to RGB if grayscale
        if len(img_display.shape) == 2:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        elif img_display.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        print(f"[DEBUG] After color conversion: {img_display.shape}")
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img_display.shape[1], img_display.shape[0]))
        print(f"[DEBUG] CAM resized: {cam_resized.shape}, min: {cam_resized.min()}, max: {cam_resized.max()}")
        
        # Normalize CAM for colormap (0-255)
        cam_resized = (cam_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        print(f"[DEBUG] Heatmap created: {heatmap.shape}")
        
        # Create overlay (60% original, 40% heatmap)
        overlay = cv2.addWeighted(img_display, 0.6, heatmap, 0.4, 0)
        print(f"[DEBUG] Overlay created: {overlay.shape}")
        
        return overlay, heatmap, cam_resized
        
    except Exception as e:
        print(f"[DEBUG] ERROR in create_heatmap_overlay: {e}")
        import traceback
        traceback.print_exc()
        raise

def array_to_base64(img_array):
    """Convert numpy array to base64 string for HTML display"""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert BGR to RGB for proper display in HTML
        img_rgb = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
    else:
        img = Image.fromarray(img_array.astype(np.uint8))
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    return img_base64

@app.route('/')
def home():
  return render_template("index.html", available_models=get_available_models())

@app.route('/about')
def about():
  return render_template("about.html")

@app.route('/research')
def research():
  return render_template("research.html")

@app.route('/api/models')
def get_models():
    """API endpoint to get available models."""
    return jsonify(get_available_models())

@app.route('/result', methods = ['POST'])
def result():
    prediction = ''
    heatmap_img = None
    overlay_img = None
    confidence = 0
    model_used = 'simple'
    model_accuracy = None
    
    if request.method == 'POST':
        # Get the selected model (default to 'simple')
        model_used = request.form.get('model', 'simple').lower()
        
        img = request.files['pic']
        img_arr = image_preprocess(img)
        result_pred = ValuePredictor(img_arr, model_name=model_used)
        
        # Get confidence score
        confidence = float(np.max(result_pred)) * 100
        
        # Get prediction
        result_class = int(np.argmax(result_pred))
        print("result from model", result_pred) 
        print("result actual", result_class) 
        
        if result_class == 0:
            prediction = 'This cell is most likely to be Not Infected with Malarial Parasite.'
            pred_label = "Uninfected"
        else:
            prediction = 'This cell is most likely to be Infected with Malarial Parasite.'
            pred_label = "Infected"
        
        # Get model accuracy
        model_accuracy = get_model_accuracy(model_used)
        print(f"[DEBUG] Retrieved model accuracy for {model_used}: {model_accuracy}")
        
        # Generate Grad-CAM
        print("\n" + "="*60)
        print("[DEBUG] STARTING GRAD-CAM GENERATION")
        print("="*60)
        try:
            print("[DEBUG] Calling generate_grad_cam...")
            cam = generate_grad_cam(current_model, img_arr, result_class)
            print(f"[DEBUG] CAM result: {cam is not None}")
            
            if cam is not None:
                print("[DEBUG] Creating heatmap overlay...")
                overlay, heatmap, cam_resized = create_heatmap_overlay(None, cam, img_arr)
                
                print(f"[DEBUG] Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")
                print(f"[DEBUG] Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
                
                # Convert to base64
                print("[DEBUG] Converting to base64...")
                heatmap_img = array_to_base64(overlay)
                overlay_img = array_to_base64(heatmap)
                
                print(f"[DEBUG] heatmap_img created: {heatmap_img is not None}, length: {len(heatmap_img) if heatmap_img else 0}")
                print(f"[DEBUG] overlay_img created: {overlay_img is not None}, length: {len(overlay_img) if overlay_img else 0}")
                print("[DEBUG] ✅✅✅ GRAD-CAM SUCCESSFUL! ✅✅✅")
            else:
                print("[DEBUG] ❌ CAM is None - gradient computation failed")
        except Exception as e:
            print(f"[DEBUG] ❌ EXCEPTION in Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
        print("="*60)
        print("\n")
        
        # Get model display name
        model_display = MODEL_NAMES_DISPLAY.get(model_used, 'Simple CNN')
        if model_used == 'simple':
            model_display = 'Simple CNN'
        
        print(prediction)
        return render_template("result.html", 
                             prediction=prediction, 
                             pred_label=pred_label,
                             confidence=round(confidence, 2),
                             heatmap_img=heatmap_img,
                             overlay_img=overlay_img,
                             model_used=model_display,
                             model_accuracy=round(model_accuracy, 2))

if __name__ == "__main__":
  app.run(debug=True)
