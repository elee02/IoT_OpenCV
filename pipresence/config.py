import argparse

class Config:
    def __init__(self):
        # Paths to models
        self.yolo_model_path = 'data/models/yolov8n.onnx'
        self.mobilefacenet_model_path = 'data/models/mobilefacenet.onnx'
        
        # Directories for input and output
        self.input_directory = 'data/test_images'
        self.output_directory = 'data/recognized_images'
        
        # Detection and recognition thresholds
        self.detection_threshold = 0.4
        self.recognition_threshold = 0.6
        
        # Image preprocessing configurations
        self.image_size = (640, 640)
        self.face_image_size = (112, 112)
        
        # Path to embeddings file
        self.embeddings_file = 'data/encodings/face_embeddings.pkl'
        
        # Other configurations
        self.classes_file = 'coco8.yaml'
        self.log_level = 'INFO'

    def parse_args(self):
        parser = argparse.ArgumentParser(description='pipresence CLI tool')
        parser.add_argument('--input', type=str, help='Input directory for images')
        parser.add_argument('--output', type=str, help='Output directory for recognized faces')
        parser.add_argument('--camera', action='store_true', help='Use device camera for real-time recognition')
        args = parser.parse_args()
        
        if args.input:
            self.input_directory = args.input
        if args.output:
            self.output_directory = args.output
        else:
            self.output_directory = self.input_directory

    def display_config(self):
        print(f"YOLO Model Path: {self.yolo_model_path}")
        print(f"MobileFaceNet Model Path: {self.mobilefacenet_model_path}")
        print(f"Input Directory: {self.input_directory}")
        print(f"Output Directory: {self.output_directory}")
        print(f"Detection Threshold: {self.detection_threshold}")
        print(f"Recognition Threshold: {self.recognition_threshold}")
        print(f"Image Size: {self.image_size}")
        print(f"Face Image Size: {self.face_image_size}")
        print(f"Classes File: {self.classes_file}")
        print(f"Log Level: {self.log_level}")
        print(f"Embeddings File: {self.embeddings_file}")