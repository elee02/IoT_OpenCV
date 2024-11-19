# This script will handle preprocessing of images to ensure consistent quality and reduce corrupted data.

# pipresence/preprocess.py
import cv2
import os
import numpy as np
import pickle
from pipresence.config import Config
from pipresence.tools.utils import (
    contains_one_person,
    extract_face
)
from pipresence.detect_faces import FaceDetector
from pipresence.recognize_faces import FaceRecognizer

class ImagePreprocessor(Config):
    def __init__(self, input_directory=None, output_directory=None):
        super().__init__()
        self.input_directory = input_directory or self.input_directory
        self.output_directory = output_directory or self.output_directory
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()

    def process_input_image(self, image_path):
        """Process a single input image and return face detection if exactly one face is found"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return None, None
        elif len(image[0]) < 640 or len(image[1]) < 640:
            print(f"[ERROR] At least one dimension is smaller than 640")
            return None, None
        # Detect faces
        detections = self.detector.detect_faces(image)
        
        if not contains_one_person(detections):
            return -1
        face_image = extract_face(image, detections)
        return face_image

    def process_database_images(self):
        """Process all images in the database structure"""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        processed_count = 0
        error_count = 0
        database = {}
        
        # Iterate through person directories
        for person_name in os.listdir(self.input_directory):
            person_input_path = os.path.join(self.input_directory, person_name)
            person_output_path = os.path.join(self.output_directory, person_name)
            embeddings = []

            if not os.path.isdir(person_input_path):
                continue
            
            # Create output directory for this person
            if not os.path.exists(person_output_path):
                os.makedirs(person_output_path)
            
            # Process each pose (left, front, right)
            for pose in ['left', 'front', 'right']:
                input_image_path = os.path.join(person_input_path, f"{pose}.jpg")
                output_image_path = os.path.join(person_output_path, f"{pose}.jpg")
                
                if not os.path.exists(input_image_path):
                    print(f"[WARNING] Missing {pose} image for {person_name}")
                    continue
                    
                print(f"[INFO] Processing {input_image_path}")
                
                # Process the image
                face_image = self.process_input_image(input_image_path)
                
                if face_image is not None:
                    # Resize face image to model input size
                    face_resized = cv2.resize(face_image, self.face_image_size)  # Standard size for most face recognition models
                    embedding = self.recognizer.recognize_face(face_resized)
                    embeddings.append(embedding)
                    # Save the processed face
                    cv2.imwrite(output_image_path, face_resized)
                    print(f"[INFO] Saved processed face to {output_image_path}")
                    processed_count += 1
                else:
                    print(f"[ERROR] Failed to process {input_image_path}")
                    error_count += 1
            if embeddings:
                # Average the embeddings from different profiles to get a more robust representation
                average_embedding = np.mean(embeddings, axis=0)
                database[person_name] = average_embedding
                print(f"[INFO] Added {person_name} to the known faces database")
        # Save embeddings to a file for future use
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(database, f)
            print(f"[INFO] Saved known face embeddings to {self.embeddings_file}")
        return processed_count, error_count
