# This script will handle face detection using YOLOv8n with ONNX Runtime.

# pipresence/detect_faces.py
import onnxruntime as ort
import numpy as np
import cv2

class FaceDetector:
    def __init__(self, model_path='yolov8n.onnx'):
        # Load YOLOv8 nano model with ONNX Runtime
        print(f"[INFO] Loading YOLOv8n model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        # Get the input name for the ONNX model
        self.input_name = self.session.get_inputs()[0].name

    def detect_faces(self, image):
        # Preprocess image for ONNX model
        print("[INFO] Preprocessing image for face detection")
        # Resize image to the required size for YOLO model
        input_image = cv2.resize(image, (640, 640))
        # Normalize pixel values to range [0, 1]
        input_image = input_image.astype(np.float32) / 255.0
        # Change image layout to channel-first format as required by ONNX model
        input_image = np.transpose(input_image, (2, 0, 1))  # Channel first
        # Add batch dimension (needed by the model)
        input_image = np.expand_dims(input_image, axis=0)

        # Perform inference on the provided image
        print("[INFO] Running inference on the image")
        outputs = self.session.run(None, {self.input_name: input_image})
        detections = outputs[0]
        faces = []

        # Iterate over detections and extract faces based on confidence threshold
        for detection in detections:
            if detection[4] > 0.4:  # Confidence threshold
                print(f"[DEBUG] Detection confidence: {detection[4]} - Adding face to the list")
                # Extract bounding box coordinates
                x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                # Crop the detected face from the original image
                face = image[y1:y2, x1:x2]
                faces.append(face)
            else:
                print(f"[DEBUG] Detection confidence too low: {detection[4]} - Ignoring")
        
        return faces, detections  # Return detected faces and detection data
