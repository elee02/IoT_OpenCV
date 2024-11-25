import logging


class Config:
    # Paths to models
    yolo_model_path = 'data/models/yolov8n-face.onnx'
    mobilefacenet_model_path = 'data/models/mobilefacenet_fixed.onnx'

    # Directories for input and output
    input_directory = 'data/images'
    output_directory = 'data/known_faces'
    
    # Detection and recognition thresholds
    detection_threshold = 0.25
    # Recommended - 0.75
    recognition_threshold = 0.75  
    
    # Image preprocessing configurations
    image_size = (640, 640)
    face_image_size = (112, 112)
    
    # Path to embeddings file
    integration_method="weighted"
    embeddings_file = 'data/encodings/face_embeddings.pkl'
    
    # Logging
    verbose = False
    log_level = logging.WARNING

    # Setup logging
    logger = logging.getLogger('pipresence')
    logger.setLevel(log_level)
    
    # Create console handler with formatting
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger if it doesn't already have handlers
    if not logger.handlers:
        logger.addHandler(ch)

    @classmethod
    def update_config(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                cls.logger.info(f"Updating the variable: Config.{key}={value}")
                setattr(cls, key, value)
            else:
                cls.logger.warning(f"{key} is not a valid attribute of Config")
    
    @classmethod
    def set_verbose(cls, verbose):
        cls.verbose = True
        """Update logging level based on verbose flag"""
        level = logging.DEBUG if verbose else logging.WARNING
        cls.logger.setLevel(level)
        for handler in cls.logger.handlers:
            handler.setLevel(level)

    @classmethod
    def display_config(cls):
        cls.logger.info(f"YOLO Model Path: {cls.yolo_model_path}")
        cls.logger.info(f"MobileFaceNet Model Path: {cls.mobilefacenet_model_path}")
        cls.logger.info(f"Input Directory: {cls.input_directory}")
        cls.logger.info(f"Output Directory: {cls.output_directory}")
        cls.logger.info(f"Detection Threshold: {cls.detection_threshold}")
        cls.logger.info(f"Recognition Threshold: {cls.recognition_threshold}")
        cls.logger.info(f"Image Size: {cls.image_size}")
        cls.logger.info(f"Face Image Size: {cls.face_image_size}")
        cls.logger.info(f"Log Level: {cls.log_level} (10 - Debug, 30 - Warning)")
        cls.logger.info(f"Embeddings File: {cls.embeddings_file}")
        cls.logger.info(f"Multiple face integration method: {cls.integration_method}")