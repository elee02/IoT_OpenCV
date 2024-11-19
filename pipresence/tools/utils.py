import cv2
from pipresence.config import Config


logger = Config.logger


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


def contains_one_person(detections: list[dict]) -> bool:
    # Check if any faces were detected
    if len(detections) == 0:
        logger.warning(f"No faces found from the camera feed")
        return False
    # If multiple faces found, select the one with highest confidence
    elif len(detections) > 1:
        logger.info(f"Multiple faces ({len(detections)}) found in the camera feed, selecting highest confidence detection")
        # Sort detections by confidence and take the highest
        detection = max(detections, key=lambda x: x["confidence"])
        logger.info(f"Selected face with confidence: {detection['confidence']:.3f}")
    return True


def extract_face(image, detections):
    detection = detections[0]
    x = round(detection["box"][0] * detection["scale"])
    y = round(detection["box"][1] * detection["scale"])
    x_plus_w = round((detection["box"][0] + detection["box"][2]) * detection["scale"])
    y_plus_h = round((detection["box"][1] + detection["box"][3]) * detection["scale"])
    return image[y: y_plus_h, x: x_plus_w]
