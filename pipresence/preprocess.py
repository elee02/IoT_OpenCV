# This script will handle preprocessing of images to ensure consistent quality and reduce corrupted data.

# pipresence/preprocess.py
import cv2
import numpy as np
import os
import pickle
from pipresence.config import Config

class ImagePreprocessor(Config):
    def __init__(self):
        super().__init__()

    def preprocess_known_faces(self, input_directory=None, output_directory=None):
        input_directory = input_directory or self.input_directory
        output_directory = output_directory or self.output_directory
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Iterate over each person's folder in the input directory
        for person_name in os.listdir(input_directory):
            person_path = os.path.join(input_directory, person_name)
            if os.path.isdir(person_path):
                print(f"[INFO] Preprocessing known faces for: {person_name}")
                # Create output sub-directory for the person if it doesn't exist
                person_output_path = os.path.join(output_directory, person_name)
                if not os.path.exists(person_output_path):
                    os.makedirs(person_output_path)
                
                # Process each profile image: left, front, right
                for profile in ['left', 'front', 'right']:
                    image_path = os.path.join(person_path, f"{profile}.jpg")
                    if os.path.exists(image_path):
                        try:
                            # Load image from the specified path
                            print(f"[INFO] Loading image from {image_path}")
                            image = cv2.imread(image_path)
                            if image is None:
                                raise ValueError(f"[ERROR] Unable to read image at {image_path}")
                            
                            # Resize image to a standard size
                            print(f"[INFO] Resizing image to {self.image_size}")
                            resized_image = cv2.resize(image, self.image_size)
                            
                            # Normalize the image values to [0, 1]
                            print("[INFO] Normalizing image values to [0, 1]")
                            normalized_image = resized_image.astype(np.float32) / 255.0
                            
                            # Save the processed image to the output directory
                            output_image_path = os.path.join(person_output_path, f"{profile}.jpg")
                            cv2.imwrite(output_image_path, normalized_image * 255)
                            print(f"[INFO] Saved preprocessed image to {output_image_path}")
                        except ValueError as e:
                            print(e)
                    else:
                        print(f"[WARNING] Missing {profile}.jpg for {person_name}, skipping this profile.")
