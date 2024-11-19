# This script will handle face recognition using MobileFaceNet model.

# pipresence/recognize_faces.py
from pipresence.config import Config
import numpy as np
import onnxruntime as ort
import cv2

class FaceRecognizer(Config):
    def __init__(self, model_path=None):
        super().__init__()
        self.mobilefacenet_model_path = model_path or self.mobilefacenet_model_path
        print(f"[INFO] Loading MobileFaceNet model from {self.mobilefacenet_model_path}")
        self.session = ort.InferenceSession(self.mobilefacenet_model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_image):
        """Preprocess face image for the MobileFaceNet model"""
        if face_image is None:
            raise ValueError("Input face image is None")
        
        # Ensure image is BGR (OpenCV default)
        if len(face_image.shape) == 2:  # Grayscale
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

        # Resize to model's required size
        face_resized = cv2.resize(face_image, self.face_image_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] range (MobileFaceNet requirement)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        
        # Transpose to NCHW format (batch, channels, height, width)
        face_transposed = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        face_batch = np.expand_dims(face_transposed, axis=0)
        
        return face_batch

    def recognize_face(self, image):
        """Generate face embedding from image"""
        try:
            # Preprocess the face image
            preprocessed_face = self.preprocess(image)
            
            # Run inference
            embedding = self.session.run(None, {self.input_name: preprocessed_face})[0]
            
            # Post-process the embedding
            # Flatten and normalize the embedding vector
            embedding_flat = embedding.flatten()
            embedding_normalized = embedding_flat / np.linalg.norm(embedding_flat)
            
            return embedding_normalized
            
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {str(e)}")
            return None

    def compare_embeddings(self, embedding1, embedding2):
        """Compare two face embeddings using cosine similarity"""
        if embedding1 is None or embedding2 is None:
            return False
        
        try:
            # Ensure embeddings are normalized
            embedding1_normalized = embedding1 / np.linalg.norm(embedding1)
            embedding2_normalized = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_normalized, embedding2_normalized)
            
            print(f"[DEBUG] Similarity score: {similarity:.3f}")
            return similarity > self.recognition_threshold
            
        except Exception as e:
            print(f"[ERROR] Embedding comparison failed: {str(e)}")
            return False
    
    def annotate_recognized(self, image, detected_face, database):
        # Recognize detected face
        print("[INFO] Recognizing detected face")
        embedding = self.recognize_face(detected_face)
        recognized = False
        # Compare detected face with known faces in the database
        for name, known_embedding in database.items():
            if self.compare_embeddings(embedding, known_embedding):
                print(f"[INFO] Recognized {name}")
                # Annotate the recognized face in the video feed
                cv2.putText(image, f"{name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                recognized = True
                break

        if not recognized:
            print("[INFO] Face not recognized, marking as Unknown")
            # Annotate the unrecognized face in the image
            cv2.putText(image, "Unknown", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)