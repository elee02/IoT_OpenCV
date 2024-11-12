Block Tentative Diagram for Skin Cancer Detection System
========================================================

@startuml
skinparam monochrome true

package "Subsystem" {
    node "Raspberry Pi" {
        [USB Interface]
        [Power Supply]
        [GPIO Interface]

    node "Camera" {
        }
    }
}
@enduml

@startuml
skinparam monochrome true
actor "User" as user

package "Machine Learning Pipeline" {
    node "Raspberry Pi OS" {
        node "Image Capturing Module" {
            frame "Camera" {
                [1. Capture image]
            }
            [2. Resize/scale image]

    }
        node "Preprocessing Module" {
            [1. Image normalization]
    [2. Data Augmentation (flip, rotate, etc.) (Optional)]
            [3. Prepare image tensor]
        }
        node "Model Inference Module" {
            [1. Load trained model]
            [2. Run inference]
            [3. Get classification result]
        }
        node "Postprocessing Module" {
            [1. Parse & interpret results]
            [2. (Optional) Show certainty]
            [3. Generate report (log)]
        }
        user ~~[Display (Optional)]
    }
}
[1. Capture image] --> [2. Resize/scale image]
[2. Resize/scale image] ..> [1. Image normalization]
[1. Image normalization] --> [2. Data Augmentation (flip, rotate, etc.) (Optional)]
[2. Data Augmentation (flip, rotate, etc.) (Optional)] --> [3. Prepare image tensor]
[3. Prepare image tensor] ..> [1. Load trained model]
[1. Load trained model] --> [2. Run inference]
[2. Run inference] --> [3. Get classification result]
[3. Get classification result] ..> [1. Parse & interpret results]
[1. Parse & interpret results] --> [2. (Optional) Show certainty]
[2. (Optional) Show certainty] --> [3. Generate report (log)]
[3. Generate report (log)] ..> [Display (Optional)]

@enduml
