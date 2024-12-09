# detect_faces.py with changes for YOLOv8n output handling and integration with the pipresence project
import numpy as np
import cv2
import cv2.dnn
import numpy as np
import time

from pipresence.config import Config

class FaceDetector(Config):
    
    detection_times = []

    def __init__(self, model_path=None):
        super().__init__()
        self.yolo_model_path = model_path or self.yolo_model_path
        # Load YOLOv8 nano model with ONNX Runtime
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(self.yolo_model_path)
        self.logger.info(f"Loading YOLOv8n face-model from {self.yolo_model_path}")
        
    def detect_faces(self, original_image):
        """
        Main function to perform inference, draw bounding boxes.

        Args:
            original_image (numpy.ndarray): ndarray of the input image.

        Returns:
            list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
        """
        start_time = time.time()
        # Read the input image
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / self.image_size[0]

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=self.image_size, swapRB=True)
        self.model.setInput(blob)

        # Perform inference
        outputs = self.model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            # if maxScore >= 0.25:
            if maxScore >= self.detection_threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            x = round(box[0] * scale)
            y = round(box[1] * scale)
            x_plus_w = round((box[0] + box[2]) * scale)
            y_plus_h = round((box[1] + box[3]) * scale)
            detection = {
                "class_id": class_ids[index],
                "class_name": "person",
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)

        end_time = time.time()  # End timer
        detection_time = end_time - start_time
        self.detection_times.append(detection_time)
        self.logger.info(f"Face detection took {detection_time:.4f} seconds")

        # Calculate mean detection time
        mean_detection_time = np.mean(self.detection_times)
        self.logger.info(f"Mean face detection time: {mean_detection_time:.4f} seconds")

        return detections
