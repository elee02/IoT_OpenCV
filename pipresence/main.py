# This script will represent the main workflow for face detection and recognition.

# pipresence/main.py
import cv2
from pipresence.detect_faces import FaceDetector
from pipresence.recognize_faces import FaceRecognizer
from pipresence.preprocess import ImagePreprocessor
from pipresence.tools.utils import (
    contains_one_person,
    load_database,
    have_dirs
)
import os
import click
from pipresence.config import Config

@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--infer', is_flag=True, help='Run inference on the camera feed or `input_path`')
@click.option('--camera', is_flag=True, help='Use device camera for real-time recognition. If not given, inference on the `input_path` is automatically chosen.')
@click.option('--encode', is_flag=True, help='Encode the preprocessed images in the given directory')
@click.option('--input-dir',type=str, help='Input directory for images that have the structured raw face images.')
@click.option('--output-dir', type=str, help='Output directory to save the detected faces')
def main(verbose, infer, camera, encode, input_dir, output_dir,):
    logger = Config.logger
    Config.set_verbose(verbose=verbose)
    logger.info(f"verbose={verbose}, camera={camera}, encode={encode}, input_dir={input_dir}, output_dir={output_dir}")
    Config.update_config(
            input_directory = input_dir,
            output_directory = output_dir
        )

    Config.display_config()
    # Initialize face detection and recognition models
    logger.info("Initializing models for face detection and recognition")
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if encode:
        if not have_dirs(input_dir, output_dir):
            return -1
        preprocessor = ImagePreprocessor()
        success, fail = preprocessor.process_database_images()
        logger.info(f"Total processed images: {success}")
        logger.info(f"Total failed processes: {fail}")
    elif infer:
        database = load_database()
        
        if camera:
            # Start the camera feed to detect and recognize faces
            logger.info("Starting camera feed for face detection and recognition")
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame from camera, exiting loop")
                    break

                # Detect faces in the current frame
                logger.info("Detecting faces in the current frame")
                detections = detector.detect_faces(frame)
                contains, detection = contains_one_person(detections)
                if not contains:
                    break

                recognizer.annotate_recognized(frame, detection, database)

                # Display the video feed with annotations
                cv2.imshow('PiPresence - Attendance Recognition', frame)

                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("'q' pressed, exiting the application")
                    break

            # Release the camera and close all windows
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera feed closed, application terminated")
        else:
            if not have_dirs(input_dir, output_dir):
                return -1
            images = os.listdir(Config.input_directory)
            # Create output directory for recognized images
            if not os.path.exists(Config.output_directory):
                os.makedirs(Config.output_directory)

            for image in images:
                img = cv2.imread(os.path.join(Config.input_directory, image))
                detections = detector.detect_faces(img)
                contains, detection = contains_one_person(detections)
                if not contains:
                    break
                
                recognizer.annotate_recognized(img, detection, database)

                # Save the recognized face
                cv2.imwrite(os.path.join(Config.output_directory, f"[id]{image}"), img)
                logger.info(f"Saved recognized face to {Config.output_directory}")
    else:
        logger.critical(f"Please, provide the options!")

if __name__ == '__main__':
    main()
