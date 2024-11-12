# This script will handle face recognition using MobileFaceNet model.

# pipresence/recognize_faces.py
import numpy as np
import onnxruntime as ort
import cv2

class FaceRecognizer:
    def __init__(self, model_path='mobilefacenet.onnx'):
        # Load MobileFaceNet ONNX model
        print(f"[INFO] Loading MobileFaceNet model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        # Get the input name for the ONNX model
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_image):
        # Resize the face image to the required input size for the model
        print("[INFO] Preprocessing face image for recognition")
        face_resized = cv2.resize(face_image, (112, 112))
        # Normalize pixel values to range [0, 1]
        face_resized = face_resized.astype(np.float32) / 255.0
        # Change image layout to channel-first format as required by ONNX model
        face_resized = np.transpose(face_resized, (2, 0, 1))  # Channel first
        # Add batch dimension (needed by the model)
        face_resized = np.expand_dims(face_resized, axis=0)
        return face_resized

    def recognize_face(self, face_image):
        # Preprocess and run inference to generate face embedding
        print("[INFO] Generating face embedding")
        preprocessed_face = self.preprocess(face_image)
        # Run inference to get the embedding
        embedding = self.session.run(None, {self.input_name: preprocessed_face})[0]
        return embedding

    def compare_embeddings(self, embedding1, embedding2, threshold=0.6):
        # Calculate similarity between two embeddings using cosine similarity
        cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"[DEBUG] Cosine similarity: {cosine_similarity}")
        # If similarity exceeds the threshold, return True (match found)
        return cosine_similarity > threshold
