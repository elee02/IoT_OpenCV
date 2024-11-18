# This script will handle preprocessing of images to ensure consistent quality and reduce corrupted data.

# pipresence/preprocess.py
import cv2
import os
from pipresence.config import Config
from pipresence.detect_faces import FaceDetector

class ImagePreprocessor(Config):
    def __init__(self, input_directory=None, output_directory=None):
        super().__init__()
        self.input_directory = input_directory or self.input_directory
        self.output_directory = output_directory or self.output_directory
        self.detector = FaceDetector()

    def process_input_image(self, image_path):
        """Process a single input image and return face detection if exactly one face is found"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return None, None
        
        # Detect faces
        detections = self.detector.detect_faces(image)
        
        # Check if exactly one face is detected
        if len(detections) != 1:
            print(f"[WARNING] Expected 1 face, found {len(detections)} in {image_path}")
            return None, None
        
        # Get the single detection
        detection = detections[0]
        
        # Extract face coordinates
        x = round(detection["box"][0] * detection["scale"])
        y = round(detection["box"][1] * detection["scale"])
        x_plus_w = round((detection["box"][0] + detection["box"][2]) * detection["scale"])
        y_plus_h = round((detection["box"][1] + detection["box"][3]) * detection["scale"])
        
        # Extract and return the face region
        face_image = image[y:y_plus_h, x:x_plus_w]
        return face_image, detection["confidence"]

    def process_database_images(self, input_dir, output_dir):
        """Process all images in the database structure"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        processed_count = 0
        error_count = 0
        
        # Iterate through person directories
        for person_name in os.listdir(input_dir):
            person_input_path = os.path.join(input_dir, person_name)
            person_output_path = os.path.join(output_dir, person_name)
            
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
                face_image, confidence = self.process_input_image(input_image_path, detector)
                
                if face_image is not None:
                    # # Resize face image to model input size
                    # face_resized = cv2.resize(face_image, (112, 112))  # Standard size for most face recognition models
                    
                    # Save the processed face
                    cv2.imwrite(output_image_path, face_image)
                    print(f"[INFO] Saved processed face to {output_image_path}")
                    processed_count += 1
                else:
                    print(f"[ERROR] Failed to process {input_image_path}")
                    error_count += 1
        
        return processed_count, error_count
