import cv2
import numpy as np
import tensorflow as tf
import base64
import io
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def build_snn_model(input_shape):
    """
    Builds a Siamese Neural Network for image comparison.
    """
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (10, 10), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4, 4), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    return Model(input_layer, x)

def decode_base64_to_image(base64_string):
    """
    Decodes a base64 string into an OpenCV image.
    """
    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def preprocess_image(image, target_size):
    """
    Prepares an image for model input.
    """
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.astype('float32') / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

def classify_similarity(distance):
    """
    Classifies similarity based on Euclidean distance.
    """
    if distance <= 0.001:
        return "Identical"
    elif distance <= 0.05:
        return "Slightly Different"
    else:
        return "Non-Identical"

def highlight_differences(image1, image2):
    """
    Highlights differences between two images and returns a base64 string of the output image.
    """
    diff = cv2.absdiff(image1, image2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Draw contours around differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Minimum size to highlight
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Encode result image to base64
    _, buffer = cv2.imencode('.png', image1)
    return base64.b64encode(buffer).decode("utf-8")

def compare_images_with_snn(base64_img1, base64_img2, model, input_size):
    """
    Compares two base64-encoded images using SNN and highlights differences.
    """
    try:
        # Decode base64 images
        image1 = decode_base64_to_image(base64_img1)
        image2 = decode_base64_to_image(base64_img2)

        # Ensure images are the same size
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # Preprocess images for model
        img1 = preprocess_image(image1, input_size)
        img2 = preprocess_image(image2, input_size)

        # Extract features using the model
        featsA = model.predict(img1)
        featsB = model.predict(img2)

        # Compute Euclidean distance
        distance = np.linalg.norm(featsA - featsB)
        similarity_category = classify_similarity(distance)

        # Generate highlighted difference image
        difference_image_base64 = highlight_differences(image1, image2)

        return {
            "distance": round(float(distance), 4),
            "classification": similarity_category,
            "difference_image_base64": difference_image_base64
        }
    except Exception as e:
        return {"error": str(e)}

# Define input size for the model
input_size = (128, 128)
input_shape = input_size + (3,)  # Using 3 channels for RGB

# Build and compile the SNN model
snn_model = build_snn_model(input_shape)
snn_model.compile(optimizer=Adam(), loss="binary_crossentropy")

@app.route('/compare', methods=['POST'])
def compare_images():
    """
    API endpoint to compare two base64-encoded images.
    """
    try:
        data = request.json
        base64_img1 = data.get("image1")
        base64_img2 = data.get("image2")

        if not base64_img1 or not base64_img2:
            return jsonify({"error": "Missing required parameters: image1, image2"}), 400

        result = compare_images_with_snn(base64_img1, base64_img2, snn_model, input_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    API health check endpoint.
    """
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True, threaded=True)
