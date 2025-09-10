import tensorflow as tf
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from PIL import Image

# Load your trained cats vs dogs model
model = tf.keras.models.load_model("kaggle_cnn_model.h5")

# Only two classes now
class_names = ["cat", "dog"]

def preprocess_image(image_file):
    # Resize to match your training size (64x64)
    img = Image.open(image_file).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)  # Match training shape
    return img_array

@api_view(["POST"])
def predict(request):
    if "image" not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    img_array = preprocess_image(request.FILES["image"])
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return Response({"prediction": predicted_class})
