import cv2
import os
import pickle
import numpy as np
import faiss

from pipresence.config import Config

logger = Config.logger


def compute_weighted_average_embedding(embeddings, weights):
    """
    Compute the weighted average of embeddings.
    
    Args:
        embeddings (list or numpy.ndarray): A list or array of face embeddings.
        weights (list): A list of weights corresponding to each embedding.
        
    Returns:
        numpy.ndarray: The weighted average embedding.
    """
    embeddings = np.array(embeddings)
    weights = np.array(weights).reshape(-1, 1)  # Ensure weights are column-wise
    weighted_average = np.sum(embeddings * weights, axis=0) / np.sum(weights)
    return weighted_average


def compute_distance_based_selection(embeddings, reference_embedding):
    """
    Select the embedding that is closest to the reference embedding using faiss.
    
    Args:
        embeddings (list or numpy.ndarray): A list or array of face embeddings.
        reference_embedding (numpy.ndarray): The reference embedding to compare against.
        
    Returns:
        numpy.ndarray: The embedding closest to the reference.
    """
    embeddings = np.array(embeddings).astype('float32')
    reference_embedding = np.array(reference_embedding).reshape(1, -1).astype('float32')

    # Using Faiss to create an index and search for the nearest neighbor
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
    index.add(embeddings)
    _, closest_index = index.search(reference_embedding, 1)

    return embeddings[closest_index[0][0]]


def compute_clustered_embedding(embeddings):
    """
    Cluster the embeddings and get the centroid for more robustness.
    Uses K-Means to cluster and returns the centroid.
    
    Args:
        embeddings (list or numpy.ndarray): A list or array of face embeddings for left, front, and right profiles.
    
    Returns:
        numpy.ndarray: The centroid of the clustered embeddings.
    """
        # Convert embeddings to a numpy array and make sure it is float32 (required by faiss)
    embeddings = np.array(embeddings).astype('float32')

    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided to compute the centroid.")

    if embeddings.shape[0] < 2:
        # If there's only one embedding, we return it directly
        return embeddings[0]

    # Adjust the number of clusters based on the number of embeddings
    n_clusters = min(embeddings.shape[0], 1)  # Ensure clusters are <= number of embeddings
    d = embeddings.shape[1]  # Dimensionality of the embeddings

    # Create a Faiss KMeans object
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True)
    
    # Train the KMeans model using the embeddings
    try:
        kmeans.train(embeddings)
    except RuntimeError as e:
        raise RuntimeError(f"KMeans training failed: {e}")
    
    # The centroid is in kmeans.centroids_
    centroid = kmeans.centroids[0]

    return centroid


# Main function to add a person to the database
def add_person_to_database(database, person_name, embeddings):
    """
    Add a person to the database using different methods to compute the face embedding.
    
    Args:
        database (dict): The database storing face embeddings.
        person_name (str): The name of the person being added.
        embeddings (list or numpy.ndarray): A list or array of face embeddings for left, front, and right profiles.
        method (str): The method to compute the final embedding ("clustered", "weighted", "distance_based").
    """
    method = Config.integration_method
    if method == "clustered":
        # Use K-Means clustering to find the centroid
        final_embedding = compute_clustered_embedding(embeddings)
    elif method == "weighted":
        # Weighted Average, assuming more importance to the front profile
        weights = [1, 2, 1]  # Assign higher weight to the front view
        final_embedding = compute_weighted_average_embedding(embeddings, weights)
    elif method == "distance_based":
        # Compute the average embedding and find the closest one
        average_embedding = np.mean(embeddings, axis=0)
        final_embedding = compute_distance_based_selection(embeddings, average_embedding)
    else:
        raise ValueError("Invalid method. Choose from 'clustered', 'weighted', or 'distance_based'.")
    
    # Add to the database
    database[person_name] = final_embedding
    logger.info(f"Added {person_name} to the known faces database")


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
    logger.debug(f"Bounding box at: {x}, {y}, {x_plus_w}, {y_plus_h} with confidence: {confidence}, person: {label}")
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def contains_one_person(detections: list[dict]) -> tuple[bool, dict]:
    # Check if any faces were detected
    if len(detections) == 0:
        logger.warning(f"No faces found from the camera feed")
        return False, {}
    # If multiple faces found, select the one with highest confidence
    elif len(detections) > 1:
        logger.info(f"Multiple faces ({len(detections)}) found in the camera feed, selecting highest confidence detection")
        # Sort detections by confidence and take the highest
        detection = max(
            detections, 
            key=lambda x: x["box"][2] * x["box"][3] * pow(x["scale"], 2)
        )
        logger.info(f"Selected face with area: {detection['box'][2] * detection['scale']:.3f}x{detection['box'][3] * detection['scale']:.3f}")
        return True, detection
    return True, detections[0]

def draw_detections(original_image: np.ndarray, detections: list[dict]):
    detection = detections[0]
    x = round(detection["box"][0] * detection["scale"])
    y = round(detection["box"][1] * detection["scale"])
    x_plus_w = round((detection["box"][0] + detection["box"][2]) * detection["scale"])
    y_plus_h = round((detection["box"][1] + detection["box"][3]) * detection["scale"])
    color = (0, 0, 255)
    confidence = detection["confidence"]
    label = detection["class_name"]
    draw_bounding_box(original_image, label, color, confidence, x, y, x_plus_w, y_plus_h)
    return original_image

def extract_face(image, detections):
    detection = detections[0]
    x = round(detection["box"][0] * detection["scale"])
    y = round(detection["box"][1] * detection["scale"])
    x_plus_w = round((detection["box"][0] + detection["box"][2]) * detection["scale"])
    y_plus_h = round((detection["box"][1] + detection["box"][3]) * detection["scale"])
    return image[y: y_plus_h, x: x_plus_w]


def load_database():
    if os.path.exists(Config.embeddings_file):
            # Load existing embeddings from the file
            logger.info(f"Loading known face embeddings from {Config.embeddings_file}")
            with open(Config.embeddings_file, 'rb') as f:
                return pickle.load(f)
    else:
        logger.error(f"[ERROR] No embeddings found")
        return -1
    

def have_dirs(input_dir, output_dir):
    if not input_dir:
        logger.error(f"--input-dir (structured raw images folder) is not provided")
        return False
    if not output_dir:
        logger.warning(f"Setting --output-dir to {input_dir}...")
        Config.update_config(output_directory=input_dir)
    return True