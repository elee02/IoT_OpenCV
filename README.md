# PiPresence

![PiPresence Logo](/home/el02/IoT_OpenCV/data/logo/PiPresence_readme.webp)

---

## Revolutionizing Attendance with the Power of Edge Computing

Welcome to **PiPresence**—the ultimate attendance solution designed for modern, edge-powered environments. Built on the versatile Raspberry Pi 4B, PiPresence merges the ingenuity of computer vision with seamless, on-device processing to deliver accurate and real-time attendance tracking. No servers. No delays. Just instant, reliable presence detection right where you need it.

With **PiPresence**, witness the future of attendance—automated, intelligent, and powered by the device in your hand.

---

Follow the guide below to embark on this journey of cutting-edge attendance automation. Let’s get started!

# Automatic Attendance Taking Software for Raspberry Pi 4B

## Overview

This project is a Python-based tool developed as a school project to automate attendance taking using a Raspberry Pi 4B and a camera. The software runs on the edge device, capturing images and processing them to recognize and record attendance without the need for manual input.

## Features

- **Real-time Image Capture**: Utilizes the Raspberry Pi camera module to capture images.
- **Face Recognition**: Identifies individuals using facial recognition algorithms.
- **Attendance Logging**: Records attendance data with timestamps.
- **Edge Processing**: All processing is done on the Raspberry Pi, ensuring data privacy and reducing the need for external servers.

## Hardware Requirements

- Raspberry Pi 4B
- Raspberry Pi Camera Module
- MicroSD Card (16GB or larger recommended)
- Power Supply for Raspberry Pi
- Optional: Monitor, Keyboard, and Mouse for setup

## Software Requirements

- Raspbian OS (Raspberry Pi OS)
- Python 3.x
- Required Python Libraries:
  - `opencv-python`
  - `face_recognition`
  - `numpy`
  - `Pillow`

## Installation

### 1. Set Up Raspberry Pi

- Install Raspbian OS on your MicroSD card.
- Boot up the Raspberry Pi with the camera connected.
- Enable the camera interface:

  ```bash
  sudo raspi-config
  ```

  Navigate to **Interface Options** and enable the camera.

### 2. Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade
```

### 3. Install Python and Pip

```bash
sudo apt-get install python3-pip
```

### 4. Install Required Python Libraries

```bash
pip3 install opencv-python face_recognition numpy Pillow
```

### 5. Clone the Repository

```bash
git clone <repository_url>
cd <project_directory>
```

## Configuration

- **Known Faces**: Place images of individuals to be recognized in the `known_faces` directory.
- **Settings**: Modify any configurable parameters in `config.py` (if available).

## Usage

```bash
python3 main.py
```

- The script will start the camera and begin processing.
- Attendance logs will be saved in the `logs` directory as a CSV file.

## Directory Structure

```
project_directory/
│
├── known_faces/
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
├── logs/
│   └── attendance_log.csv
├── src/
│   ├── main.py
│   ├── camera_module.py
│   ├── face_recognition_module.py
│   └── ...
├── config.py
└── README.md
```

## Troubleshooting

- **Camera Not Found**: Ensure the camera is enabled in the Raspberry Pi settings and connected properly.
- **Module Errors**: Verify that all Python libraries are installed.
- **Recognition Issues**: Make sure the known faces are clear and well-lit images.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Special thanks to the school faculty and peers for their support.
- Libraries and resources:
  - [OpenCV](https://opencv.org/)
  - [face_recognition](https://github.com/ageitgey/face_recognition)