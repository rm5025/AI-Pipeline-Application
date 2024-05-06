from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torchvision.transforms as T
import requests
import os
import io
import json
import urllib.request
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 for object detection
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def detect_objects(image):
    results = model_yolov5(image)
    data = results.pandas().xyxy[0].to_dict(orient="records")

    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    for item in data:
        # Convert bounding box coordinates to integers
        xmin, ymin, xmax, ymax = map(int, (item['xmin'], item['ymin'], item['xmax'], item['ymax']))

        # Draw bounding box
        cv2.rectangle(open_cv_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Draw label
        cv2.putText(open_cv_image, item['name'], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Convert OpenCV image back to PIL format
    image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))

    return data, image


# Load the pre-trained classification model
model_classification = models.resnet50(pretrained=True)
model_classification.eval()

# Image transformations for the classification model
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the ImageNet class labels
CLASS_INDEX = json.load(urllib.request.urlopen(
    'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'))


def classify_objects(image):
    # Convert the image to a PyTorch tensor
    image = transform(image)
    image = image.unsqueeze(0)

    # Perform inference with the classification model
    outputs = model_classification(image)
    _, preds = torch.max(outputs, 1)

    # Map the output indices to the actual class labels
    class_labels = [CLASS_INDEX[i] for i in preds.tolist()]

    return {"classes": class_labels}


# Load ViT-GPT2 model for image captioning
model_captioning = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_captioning.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


# Generate captions using ViT-GPT2
def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model_captioning.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


# Process images and return JSON response
@app.route('/process_images', methods=['GET'])
def process_images():
    # Specify the directory where the images are stored
    image_dir = 'C:\\Users\\tntra\\Downloads\\images'

    # Initialize a dictionary to store the results for all images
    all_results = {}

    # Loop over the images in the directory
    for i in range(27):
        # Open the image file
        image_path = os.path.join(image_dir, f'image-{i}.jpg')

        # Check if the image file exists
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path)

        # Perform object detection using YOLOv5
        detection_results, image_with_boxes = detect_objects(image)

        # Perform object classification
        classification_results = classify_objects(image)

        # Perform image captioning using ViT-GPT2
        captioning_results = generate_caption(image)

        # Draw caption on the image
        cv_image = np.array(image_with_boxes)
        cv_image = cv_image[:, :, ::-1].copy()
        cv2.putText(cv_image, captioning_results[0], (50, cv_image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)
        image_with_boxes = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Save the image with bounding boxes and caption
        image_with_boxes.save(os.path.join(image_dir, f'image-{i}_with_boxes_and_caption.jpg'))

        # Combine the results into a JSON response
        result = {
            'detection': detection_results,
            'classification': classification_results,
            'captioning': captioning_results
        }

        # Store the results for this image
        all_results[f'image-{i}'] = result

    return jsonify(all_results)


@app.route('/', methods=['GET'])
def home():
    return "Server is running!"


if __name__ == '__main__':
    # Run the Flask app and specify host, port, and debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)

    # Print the server URL to the console
    print(f"Server running at http://127.0.0.1:5000/")
