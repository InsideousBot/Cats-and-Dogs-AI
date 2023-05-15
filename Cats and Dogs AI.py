import sys
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Prepare the image
def prepare_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict the class of the image
def predict_image(image_path):
    image = prepare_image(image_path)
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0]

# Check if the image is a cat or a dog
def is_cat_or_dog(image_path):
    prediction = predict_image(image_path)
    label, name, confidence = prediction

    if name == "tiger_cat" or name == "Egyptian_cat" or "tabby" in name:
        return "cat"
    elif "dog" in name or "hound" in name:
        return "dog"
    else:
        return "unknown"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cat_dog_classifier.py path_to_image")
        sys.exit(1)

    image_path = sys.argv[1]
    result = is_cat_or_dog(image_path)
    print(f"The image is a: {result}")
