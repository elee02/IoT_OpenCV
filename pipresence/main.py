# This script will represent the main workflow for face detection and recognition.

# pipresence/main.py
import cv2
from pipresence.detect_faces import FaceDetector
from pipresence.recognize_faces import FaceRecognizer
from pipresence.preprocess import ImagePreprocessor
from pipresence.tools.utils import (
    encode_faces
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
    # Initialize face detection and recognition models
    print("[INFO] Initializing models for face detection and recognition")
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if encode:
        encode_faces(
            config=config, 
            recognizer=recognizer
        )

    if os.path.exists(config.embeddings_file):
        # Load existing embeddings from the file
        print(f"[INFO] Loading known face embeddings from {config.embeddings_file}")
        with open(config.embeddings_file, 'rb') as f:
            database = pickle.load(f)
    else:
        raise Exception("[ERROR] No embeddings found")
    
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
            
            for detection in detections:
                x = round(detection["box"][0] * detection["scale"])
                y = round(detection["box"][1] * detection["scale"])
                x_plus_w = round((detection["box"][0] + detection["box"][2]) * detection["scale"])
                y_plus_h = round((detection["box"][1] + detection["box"][3]) * detection["scale"])
                detected_face = frame[y: y_plus_h, x: x_plus_w]
                # Recognize detected face
                print("[INFO] Recognizing detected face")
                embedding = recognizer.recognize_face(detected_face)
                recognized = False

                # Compare detected face with known faces in the database
                for name, known_embedding in database.items():
                    if recognizer.compare_embeddings(embedding, known_embedding):
                        print(f"[INFO] Recognized {name}")
                        # Annotate the recognized face in the video feed
                        cv2.putText(frame, f"{name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        recognized = True
                        break

                if not recognized:
                    print("[INFO] Face not recognized, marking as Unknown")
                    # Annotate the unrecognized face in the video feed
                    cv2.putText(frame, "Unknown", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
        # Preprocess known faces
        print("[INFO] Starting preprocessing of known faces")
        preprocessor = ImagePreprocessor()
        preprocessor.preprocess_known_faces(config.input_directory, preprocessor.output_directory)

        

if __name__ == '__main__':
    main()
