# This script will represent the main workflow for face detection and recognition.

# pipresence/main.py
import cv2
from pipresence.detect_faces import FaceDetector
from pipresence.recognize_faces import FaceRecognizer
from pipresence.preprocess import ImagePreprocessor
from pipresence.tools.utils import (
    contains_one_person,
    extract_face
)
import numpy as np
import os
import pickle
import click
from pipresence.config import Config

@click.command()
@click.option('--infer', is_flag=True, help='Run inference on the camera feed or `input_path`')
@click.option('--camera', is_flag=True, help='Use device camera for real-time recognition. If not given, inference on the `input_path` is automatically chosen.')
@click.option('--encode', is_flag=True, help='Encode the preprocessed images in the given directory')
@click.option('--input-dir', default='data/images',type=str, help='Input directory for images that have the structured raw face images.')
@click.option('--output-dir', default='data/known_faces', type=str, help='Output directory to save the detected faces')
def main(infer, camera, encode, input_dir, output_dir,):
    Config.update_config(
            input_directory = input_dir,
            output_directory = output_dir
        )
    # Initialize face detection and recognition models
    print("[INFO] Initializing models for face detection and recognition")
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if encode:
        preprocessor = ImagePreprocessor()
        success, fail = preprocessor.process_database_images()
        print(f"[INFO] Total processed images: {success}")
        print(f"[INFO] Total failed processes: {fail}")
    if infer:
        if os.path.exists(Config.embeddings_file):
            # Load existing embeddings from the file
            print(f"[INFO] Loading known face embeddings from {Config.embeddings_file}")
            with open(Config.embeddings_file, 'rb') as f:
                database = pickle.load(f)
        else:
            print(f"[ERROR] No embeddings found")
            return -1
        
        if camera:
            # Start the camera feed to detect and recognize faces
            print("[INFO] Starting camera feed for face detection and recognition")
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame from camera, exiting loop")
                    break

                # Detect faces in the current frame
                print("[INFO] Detecting faces in the current frame")
                detections = detector.detect_faces(frame)
                
                if not contains_one_person(detections):
                    break

                recognizer.annotate_recognized(frame, detections, database)

                # Display the video feed with annotations
                cv2.imshow('PiPresence - Attendance Recognition', frame)

                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] 'q' pressed, exiting the application")
                    break

            # Release the camera and close all windows
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera feed closed, application terminated")
        else:
            images = os.listdir(Config.input_directory)
            # Create output directory for recognized images
            if not os.path.exists(Config.output_directory):
                os.makedirs(Config.output_directory)

            for image in images:
                img = cv2.imread(os.path.join(Config.input_directory, image))
                detections = detector.detect_faces(img)

                if not contains_one_person(detections):
                    break
                
                recognizer.annotate_recognized(img, detections, database)

                # Save the recognized face
                cv2.imwrite(os.path.join(Config.output_directory, image), img)
                print(f"[INFO] Saved recognized face to {Config.output_directory}")

if __name__ == '__main__':
    main()
