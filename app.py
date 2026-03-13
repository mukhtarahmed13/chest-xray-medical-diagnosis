from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import cv2
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Define weighted loss function
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += -tf.reduce_mean(pos_weights[i] * y_true[:, i] * tf.math.log(y_pred[:, i] + epsilon) 
                                   + neg_weights[i] * (1 - y_true[:, i]) * tf.math.log(1 - y_pred[:, i] + epsilon))
        return loss
    return weighted_loss

# Load model
model = load_model('best_model.keras', compile=False)

# Define labels
labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 
          'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 
          'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

def preprocess_image(img):
    """Preprocess image for model input"""
    img = img.resize((320, 320))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize
    img_array = (img_array - img_array.mean()) / img_array.std()
    return img_array

def generate_gradcam(img_array, class_idx, layer_name='conv5_block16_concat'):
    """Generate Grad-CAM heatmap"""
    # Get the target layer
    try:
        conv_layer = model.get_layer(layer_name)
    except:
        # Fallback to last convolutional layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                conv_layer = layer
                break
    
    # Create a model that outputs both predictions and conv layer output
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.output, conv_layer.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        model_out, conv_out = grad_model(img_array)
        class_channel = model_out[:, class_idx]
    
    # Get gradients of the class with respect to the conv layer
    grads = tape.gradient(class_channel, conv_out)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradients
    conv_out = conv_out[0]
    pooled_grads = pooled_grads.numpy()
    conv_out = conv_out.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_out[:, :, i] *= pooled_grads[i]
    
    # Average over all channels
    heatmap = np.mean(conv_out, axis=-1)
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-10)
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (320, 320))
    
    return heatmap

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    # Convert to uint8
    img = img_array[0]
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    img = (img * 255).astype(np.uint8)
    
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Overlay
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlayed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        threshold = float(request.form.get('threshold', 0.5))
        
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img_array = preprocess_image(img)
        
        # Make predictions
        predictions = model.predict(img_array)[0]
        
        # Get top predictions
        results = []
        gradcams = []
        
        for i, label in enumerate(labels):
            if predictions[i] > threshold:
                results.append({
                    'disease': label,
                    'probability': float(predictions[i])
                })
                
                # Generate Grad-CAM for this class
                heatmap = generate_gradcam(img_array, i)
                overlayed = overlay_heatmap(img_array, heatmap)
                
                # Convert to base64
                _, buffer = cv2.imencode('.png', overlayed)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                gradcams.append({
                    'disease': label,
                    'image': img_base64
                })
        
        if not results:
            results.append({
                'disease': 'No significant findings',
                'probability': 1.0
            })
        
        # Sort by probability
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'predictions': results,
            'gradcams': gradcams
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
