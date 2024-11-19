class Config:
    # Paths to models
    yolo_model_path = 'data/models/yolov8n-face.onnx'
    mobilefacenet_model_path = 'data/models/mobilefacenet_fixed.onnx'
    
    # Directories for input and output
    input_directory = 'data/images'
    output_directory = 'data/known_faces'
    
    # Detection and recognition thresholds
    detection_threshold = 0.25
    recognition_threshold = 0.4
    
    # Image preprocessing configurations
    image_size = (640, 640)
    face_image_size = (112, 112)
    
    # Path to embeddings file
    embeddings_file = 'data/encodings/face_embeddings.pkl'
    
    # Other configurations
    log_level = 'INFO'

    @classmethod
    def update_config(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                print(f"[WARNING] {key} is not a valid attribute of Config")
    
    @classmethod
    def display_config(cls):
        print(f"YOLO Model Path: {cls.yolo_model_path}")
        print(f"MobileFaceNet Model Path: {cls.mobilefacenet_model_path}")
        print(f"Input Directory: {cls.input_directory}")
        print(f"Output Directory: {cls.output_directory}")
        print(f"Detection Threshold: {cls.detection_threshold}")
        print(f"Recognition Threshold: {cls.recognition_threshold}")
        print(f"Image Size: {cls.image_size}")
        print(f"Face Image Size: {cls.face_image_size}")
        print(f"Log Level: {cls.log_level}")
        print(f"Embeddings File: {cls.embeddings_file}")