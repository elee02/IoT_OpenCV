# This script will handle face recognition using MobileFaceNet model.

# pipresence/recognize_faces.py
from pipresence.tools.utils import draw_bounding_box, extract_face
from pipresence.config import Config
import numpy as np
import onnxruntime as ort
import cv2

class FaceRecognizer(Config):
    def __init__(self, model_path=None):
        super().__init__()
        self.mobilefacenet_model_path = model_path or self.mobilefacenet_model_path
        self.logger.info(f"Loading MobileFaceNet model from {self.mobilefacenet_model_path}")
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
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] range (MobileFaceNet requirement)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        
        # # Transpose to NCHW format (batch, channels, height, width)
        # face_transposed = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
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
            self.logger.error(f"Face recognition failed: {str(e)}")
            return None

    def compare_embeddings(self, embedding1, embedding2):
        """Compare two face embeddings using cosine similarity"""
        if embedding1 is None or embedding2 is None:
            return False, 0
        
        try:
            # Ensure embeddings are normalized
            embedding1_normalized = embedding1 / np.linalg.norm(embedding1)
            embedding2_normalized = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1_normalized, embedding2_normalized)
            
            self.logger.debug(f"Similarity score: {similarity:.3f}")
            return similarity > self.recognition_threshold, similarity
            
        except Exception as e:
            self.logger.error(f"Embedding comparison failed: {str(e)}")
            return False, 0
    
    def annotate_recognized(self, image, detections, database):
        detection = detections[0]
        x = round(detection["box"][0] * detection["scale"])
        y = round(detection["box"][1] * detection["scale"])
        x_plus_w = round((detection["box"][0] + detection["box"][2]) * detection["scale"])
        y_plus_h = round((detection["box"][1] + detection["box"][3]) * detection["scale"])
        detected_face = image[y: y_plus_h, x: x_plus_w]
        # Recognize detected face
        self.logger.info("Recognizing detected face")
        embedding = self.recognize_face(detected_face)
        recognized = False
        # Compare detected face with known faces in the database
        for name, known_embedding in database.items():
            matches, similarity = self.compare_embeddings(embedding, known_embedding)
            if matches:
                self.logger.info(f"Recognized {name}")
                # Annotate the recognized face in the video feed
                draw_bounding_box(
                    img = image,
                    label = f"{name}",
                    color = (0, 255, 0),
                    confidence = similarity,
                    x = x,
                    y = y,
                    x_plus_w = x_plus_w,
                    y_plus_h = y_plus_h
                )
                recognized = True
                break

        if not recognized:
            self.logger.info("Face not recognized, marking as Unknown")
            # Annotate the unrecognized face in the image
            draw_bounding_box(
                    img = image,
                    label = f"Unknown",
                    color = (0, 255, 0),
                    confidence = 0.,
                    x = x,
                    y = y,
                    x_plus_w = x_plus_w,
                    y_plus_h = y_plus_h
                )