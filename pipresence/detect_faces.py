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
        detections = outputs[0][0]
        faces = []

        # Iterate over detections and extract faces based on confidence threshold
        for detection in detections:
            if detection[4] > 65:  # Confidence threshold
                # After detecting faces, add this to draw bounding boxes
                h, w, _ = image.shape
                x1, y1, x2, y2 = max(0, int(detection[0])), max(0, int(detection[1])), min(w, int(detection[2])), min(h, int(detection[3]))

                if x2 > x1 and y2 > y1:
                    face = image[y1:y2, x1:x2]
                    if face.size != 0:
                        print(f"[DEBUG] Detection confidence: {detection[4]} - Adding face to the list")
                        faces.append(face)
                        # Draw a rectangle on the image for visual verification
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        print(f"[DEBUG] Bounding box at: {x1}, {y1}, {x2}, {y2}")
                    else:
                        print(f"[WARNING] Cropped face is empty, skipping this detection.")
                else:
                    print(f"[WARNING] Invalid bounding box coordinates: ({x1}, {y1}, {x2}, {y2}) - Skipping this detection")
            else:
                print(f"[DEBUG] Detection confidence too low: {detection[4]} - Ignoring")
        
        return faces, detections  # Return detected faces and detection data
