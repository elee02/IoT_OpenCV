@startuml
[*] --> CLI
CLI --> CheckMode

state CheckMode {
    [*] --> ModeCheck
    ModeCheck --> Encode : --encode flag
    ModeCheck --> Inference : --infer flag
    ModeCheck --> Error : no flag
    Error --> [*]
}

state Encode {
    [*] --> LoadPreprocessor
    LoadPreprocessor --> ProcessDatabase
    
    state ProcessDatabase {
        [*] --> LoadImage
        LoadImage --> DetectFaces
        DetectFaces --> CheckFaces
        
        state CheckFaces {
            [*] --> CountFaces
            CountFaces --> SingleFace : one face
            CountFaces --> NoFace : no face
            CountFaces --> MultipleFaces : multiple faces
            NoFace --> SkipImage
            MultipleFaces --> SelectLargestFace
            SelectLargestFace --> ExtractFace
            SingleFace --> ExtractFace
        }
        
        ExtractFace --> GenerateEmbedding
        GenerateEmbedding --> SaveProcessedFace
        SaveProcessedFace --> IntegrateEmbeddings
        
        state IntegrateEmbeddings {
            [*] --> CheckMethod
            CheckMethod --> Clustered : clustered method
            CheckMethod --> Weighted : weighted method
            CheckMethod --> DistanceBased : distance method
            Clustered --> CombineEmbeddings
            Weighted --> CombineEmbeddings
            DistanceBased --> CombineEmbeddings
        }
        
        SkipImage --> NextImage
        CombineEmbeddings --> NextImage
        NextImage --> LoadImage : more images
        NextImage --> SaveDatabase : no more images
    }
    
    SaveDatabase --> [*]
}

state Inference {
    [*] --> CheckInputMode
    CheckInputMode --> CameraMode : --camera flag
    CheckInputMode --> ImageMode : no --camera flag
    
    state CameraMode {
        [*] --> InitCamera
        InitCamera --> LoadModels
        LoadModels --> CaptureFrame
        CaptureFrame --> DetectFacesLive
        DetectFacesLive --> CheckFacesLive
        
        state CheckFacesLive {
            [*] --> CountFacesLive
            CountFacesLive --> SingleFaceLive : one face
            CountFacesLive --> NoFaceLive : no face
            CountFacesLive --> MultipleFacesLive : multiple faces
            NoFaceLive --> ContinueCapture
            MultipleFacesLive --> SelectLargestFaceLive
            SelectLargestFaceLive --> RecognizeFace
            SingleFaceLive --> RecognizeFace
        }
        
        RecognizeFace --> CompareWithDatabase
        CompareWithDatabase --> DrawAnnotations
        DrawAnnotations --> DisplayFrame
        DisplayFrame --> CheckQuit
        ContinueCapture --> CaptureFrame
        CheckQuit --> CaptureFrame : continue
        CheckQuit --> CloseCamera : 'q' pressed
        CloseCamera --> [*]
    }
    
    state ImageMode {
        [*] --> LoadInputImage
        LoadInputImage --> ProcessSingleImage
        ProcessSingleImage --> SaveAnnotatedImage
        SaveAnnotatedImage --> NextInputImage
        NextInputImage --> LoadInputImage : more images
        NextInputImage --> [*] : no more images
    }
}

@enduml