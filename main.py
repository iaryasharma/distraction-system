import cv2
import mediapipe as mp
import numpy as np
import winsound  # Add winsound for playing beep sound

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# Configure MediaPipe Face Mesh and Holistic models
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
        mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process the image to detect faces using MediaPipe Face Detection
        detection_results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Adjusting the text position and size based on the bounding box
                font_scale = 0.5
                text_thickness = 1
                text_offset = 20
                text_x = x
                text_y = y - text_offset
                text_color = (0, 255, 0)

                cv2.putText(image, "Face Detected", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            text_color, text_thickness)

        # Process the image to detect face landmarks using MediaPipe Face Mesh
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)  # Specify connections directly here

                # Head pose estimation
                face_3d = []
                face_2d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
                            nose_3d = (1 * image.shape[1], lm.y * image.shape[0], lm.z * 3000)
                        x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                        # Get the 2D Coordinates
                        face_2d.append([x, y])
                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert to NumPy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * image.shape[1]
                cam_matrix = np.array([[focal_length, 0, image.shape[0] / 2],
                                       [0, focal_length, image.shape[1] / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, _ = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                    # Play beep sound and print message
                    winsound.Beep(1000, 200)  # Adjust frequency (Hz) and duration (ms) as needed
                    print("User seems to be distracted: Looking Left")
                elif y > 10:
                    text = "Looking Right"
                    # Play beep sound and print message
                    winsound.Beep(1000, 200)  # Adjust frequency (Hz) and duration (ms) as needed
                    print("User seems to be distracted: Looking Right")
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1]))

                # Draw line indicating nose direction
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                font_scale = 0.5
                text_thickness = 1
                text_offset = 20
                text_x = 20
                text_y = 50
                text_color = (0, 255, 0)
                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                            text_thickness)
                cv2.putText(image, "x: " + str(np.round(x * 10)), (text_x, text_y + text_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (text_x, text_y + 2 * text_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (text_x, text_y + 3 * text_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)

        # Display the annotated image in a window
        cv2.imshow('MediaPipe FaceMesh', image)

        # Press 'q' to exit the program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
