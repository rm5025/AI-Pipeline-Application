AI Pipeline application:
This AI pipeline application is a service that serves a simple API request and gives back the
response in JSON. The application accepts the list of images in a defined format and the output
should be the response containing inference through various AI models present as part of the AI
pipeline. The goal is to have an application's latency as minimum as possible and scale it to as
many users as possible.

AI Pipeline blocks and its documentation:
1. Flask Webserver: https://flask.palletsprojects.com/en/2.2.x/

2. Detection Model: Yolo-v5 COCO pretrained object detection model-
https://huggingface.co/spaces/nakamura196/yolov5-char/blob/0f967fc973c5b77dbe95cd

0cba1d328b14c884a1/ultralytics/yolov5/README.md
3. Classification Model: Pytorch Imagenet pre-trained classification model -
https://pytorch.org/vision/stable/models.html
4. Captioning Model: HuggingFace image captioning pre-trained transformer -
https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
