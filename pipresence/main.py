# This script will represent the main workflow for face detection and recognition.

# pipresence/main.py
import cv2
from pipresence.detect_faces import FaceDetector
from pipresence.recognize_faces import FaceRecognizer
from pipresence.preprocess import ImagePreprocessor
import numpy as np
import os
import pickle
import click

@click.command()
@click.option('--data-dir', default='data/images/', help='Directory for raw images.')
@click.option('--output-dir', default='data/known_faces/', help='Directory to store preprocessed images.')
@click.option('--embeddings-file', default='data/encodings/face_embeddings.pkl', help='Path to save/load face embeddings.')
def main(data_dir, output_dir, embeddings_file):
    # Preprocess known faces
    print("[INFO] Starting preprocessing of known faces")
    preprocessor = ImagePreprocessor()
    preprocessor.preprocess_known_faces(data_dir, output_dir)

    # Initialize face detection and recognition models
    print("[INFO] Initializing models for face detection and recognition")
    detector = FaceDetector(model_path='data/models/yolov8n.onnx')
    recognizer = FaceRecognizer(model_path='data/models/mobilefacenet.onnx')

    # Load or create the known face embeddings database
    database = {}

    if os.path.exists(embeddings_file):
        # Load existing embeddings from the file
        print(f"[INFO] Loading known face embeddings from {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            database = pickle.load(f)
    else:
        # Generate embeddings for known faces
        print("[INFO] Generating known face embeddings")
        database_path = output_dir
        for person_name in os.listdir(database_path):
            person_path = os.path.join(database_path, person_name)
            if os.path.isdir(person_path):
                embeddings = []
                # Process each profile image: left, front, right
                for profile in ['left', 'front', 'right']:
                    image_path = os.path.join(person_path, f"{profile}.jpg")
                    if os.path.exists(image_path):
                        known_image = cv2.imread(image_path)
                        if known_image is None:
                            print(f"[WARNING] Could not read image {image_path}, skipping.")
                            continue
                        # Generate embedding for the known face
                        embedding = recognizer.recognize_face(known_image)
                        embeddings.append(embedding)
                if embeddings:
                    # Average the embeddings from different profiles to get a more robust representation
                    average_embedding = np.mean(embeddings, axis=0)
                    database[person_name] = average_embedding
                    print(f"[INFO] Added {person_name} to the known faces database")
        # Save embeddings to a file for future use
        with open(embeddings_file, 'wb') as f:
            pickle.dump(database, f)
        print(f"[INFO] Saved known face embeddings to {embeddings_file}")

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
        faces, _ = detector.detect_faces(frame)
        for face in faces:
            # Recognize detected face
            print("[INFO] Recognizing detected face")
            embedding = recognizer.recognize_face(face)
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
