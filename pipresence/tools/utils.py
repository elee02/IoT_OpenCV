import cv2
import os
import pickle
import numpy as np


def draw_bounding_box(img, label, color, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{label} ({confidence:.2f})"
    print(f"[DEBUG] Bounding box at: {x}, {y}, {x_plus_w}, {y_plus_h} with confidence: {confidence}")
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def encode_faces(config, recognizer):
    # Load or create the known face embeddings database
    database = {}
    # Generate embeddings for known faces
    print("[INFO] Generating known face embeddings")
    database_path = config.output_directory
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
    with open(config.embeddings_file, 'wb') as f:
        pickle.dump(database, f)
    print(f"[INFO] Saved known face embeddings to {config.embeddings_file}")
    