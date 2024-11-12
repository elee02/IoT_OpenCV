# Face Recognition Inference Script
# This script takes the face encodings, compares them with known encodings, and annotates the image with the name of the recognized face.

import cv2
import numpy as np
import pickle
import onnxruntime as ort

# Load known encodings
with open('data/encodings/face_embeddings.pkl', 'rb') as f:
    known_encodings = pickle.load(f)

# Load face detection model (YOLOv8n)
detector_session = ort.InferenceSession('data/models/yolov8n.onnx')
detector_input_name = detector_session.get_inputs()[0].name

# Load face recognition model (MobileFaceNet)
recognizer_session = ort.InferenceSession('data/models/mobilefacenet_fixed.onnx')
recognizer_input_name = recognizer_session.get_inputs()[0].name

# Define cosine similarity function
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Preprocess the image for detection
def preprocess_image_for_detection(image):
    resized_image = cv2.resize(image, (640, 640))
    normalized_image = resized_image.astype(np.float32) / 255.0
    transposed_image = np.transpose(normalized_image, (2, 0, 1))  # Convert HWC to CHW
    input_image = np.expand_dims(transposed_image, axis=0)
    return input_image

# Preprocess the face image for recognition
def preprocess_face_for_recognition(face_image):
    resized_face = cv2.resize(face_image, (112, 112))
    normalized_face = resized_face.astype(np.float32) / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    return input_face

# Perform inference
image_path = 'data/test_images/idris_0.jpg'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load image from {image_path}")

# Preprocess image for detection
detection_input = preprocess_image_for_detection(image)

# Run YOLOv8 detection
outputs = detector_session.run(None, {detector_input_name: detection_input})
detections = outputs[0]

faces = []
bounding_boxes = []

# Iterate over detections to extract faces and bounding boxes
for detection in detections[0]:
    confidence = detection[4]
    if confidence > 0.4:  # Confidence threshold
        # These coordinates (cx, cy, w, h) are assumed to be the central point (cx, cy) of the bounding box, 
        # and w and h represent the width and height of the bounding box respectively.
        cx, cy, w, h = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
        # Calculate the top-left (x1, y1) and bottom-right (x2, y2) coordinates from the central point, width, and height.
        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(image.shape[1], cx + w // 2)
        y2 = min(image.shape[0], cy + h // 2)
        # This condition checks if the bounding box is valid (i.e., it has a positive area).
        # It ensures that x2 is greater than x1 and y2 is greater than y1, otherwise the bounding box would be invalid.
        if x2 > x1 and y2 > y1:
            # This line extracts the region of the image defined by the bounding box.
            # If the bounding box is invalid or goes beyond the image limits, this might lead to an empty or incorrect face crop.
            face = image[y1:y2, x1:x2]
            faces.append(face)
            bounding_boxes.append((x1, y1, x2, y2))

# Recognize faces in the image
for i, face in enumerate(faces):
    preprocessed_face = preprocess_face_for_recognition(face)
    embedding = recognizer_session.run(None, {recognizer_input_name: preprocessed_face})[0].flatten()

    recognized_name = "Unknown"
    highest_similarity = 0

    # Compare with known embeddings
    for name, known_embedding in known_encodings.items():
        similarity = cosine_similarity(embedding, known_embedding)
        if similarity > 0.6 and similarity > highest_similarity:
            recognized_name = name
            highest_similarity = similarity

    # Draw the bounding box and label
    x1, y1, x2, y2 = bounding_boxes[i]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, recognized_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the final image with bounding boxes and names
cv2.imshow('Recognized Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
