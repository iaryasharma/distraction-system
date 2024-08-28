# Distraction Detection System

This project is a real-time distraction detection system that uses MediaPipe and OpenCV to monitor a user's head movements. The system alerts the user if they seem distracted by looking left or right. It can be particularly useful in scenarios where maintaining focus is crucial, such as during studying or working.

## Features

- **Real-time Face Detection**: Utilizes MediaPipe Face Detection to detect faces in the video feed.
- **Face Mesh Landmarks**: Leverages MediaPipe Face Mesh to obtain detailed facial landmarks.
- **Head Pose Estimation**: Computes the head's orientation in 3D space.
- **Distraction Alerts**: Plays a beep sound if the user looks away from the center (left or right), indicating potential distraction.
- **Live Video Feed**: Displays the live video feed with annotations for face detection, face mesh, and head orientation.

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/Arya0077/distraction-system.git
    cd distraction-system
    ```

2. Install the required packages:

    ```bash
    pip install opencv-python mediapipe numpy
    ```

3. Run the program:

    ```bash
    python main.py
    ```

## Usage

- The program uses your webcam as the video input.
- It detects the face and estimates head pose to determine if the user is looking forward, left, right, up, or down.
- If the user is looking left or right, the program plays a beep sound to indicate distraction.

## Code Overview

- **Face Detection**: Detects faces in real-time and draws bounding boxes around them.
- **Face Mesh**: Detects and draws facial landmarks.
- **Head Pose Estimation**: Calculates the angles of the head to determine its orientation.
- **Distraction Alerts**: Plays a sound alert if the user looks left or right, indicating a distraction.

## Future Plans

- **Improved Detection**: Enhance the accuracy of head pose estimation, especially for subtle movements.
- **Custom Alerts**: Allow users to customize the type and frequency of alerts.
- **Multi-Person Detection**: Expand the system to simultaneously detect and monitor multiple faces.
- **Distraction Analytics**: Implement a feature to track and log distractions over time, providing insights into user focus patterns.
- **Integration with Productivity Tools**: Explore integrations with productivity tools or browsers to pause activities when the user is distracted.

## Collaborators

This project is developed and maintained by:

- [Arya](https://github.com/Arya0077)
- [Anant](https://github.com/AnantKhandaka)
- [Zubin](https://github.com/IMPULSINATOR)

<p align="center">
  <img src="https://github.com/Arya0077.png?size=50" alt="Arya0077" width="50" height="50" style="border-radius:50%; margin-right: 10px;">
  <img src="https://github.com/IMPULSINATOR.png?size=50" alt="IMPULSINATOR" width="50" height="50" style="border-radius:50%; margin-right: 10px;">
  <img src="https://github.com/AnantKhandaka.png?size=50" alt="AnantKhandaka" width="50" height="50" style="border-radius:50%; margin-right: 10px;">
</p>

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
