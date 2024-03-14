import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuring FaceMesh and drawing specifications
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Capturing video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    # Flip the image horizontally for a later selfie-view display and convert color space
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx == 1:
                        nose_2d = (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
                        nose_3d = (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]), lm.z * 3000)

                    x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * image.shape[1]
            cam_matrix = np.array([[focal_length, 0, image.shape[1] / 2],
                                   [0, focal_length, image.shape[0] / 2],
                                   [0, 0, 1]])

            # Distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotation matrix
            rot_mat, _ = cv2.Rodrigues(rot_vec)

            # Get angles
            angles = cv2.RQDecomp3x3(rot_mat)[0]

            # Get the y rotation degree
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360

            # Determine the direction of head tilt
            if y_angle < -10:
                text = "Looking Left"
                print("The user is distracted")
                winsound.Beep(1000, 500)
            elif y_angle > 10:
                text = "Looking Right"
                print("The user is distracted")
                winsound.Beep(1000, 500)
            elif x_angle < -10:
                text = "Looking Down"
            elif x_angle > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display nose direction
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1]))

            # Calculate FPS
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime

            # Draw line and add text
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, f'x: {np.round(x_angle, 2)}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'y: {np.round(y_angle, 2)}', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'z: {np.round(z_angle, 2)}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
