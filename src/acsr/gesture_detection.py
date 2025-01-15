# !pip install mediapipe opencv-python numpy
import numpy as np
import mediapipe as mp
import cv2

def eye_aspect_ratio(eye_landmarks):
    dist = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y, eye_landmarks[1].z]) - np.array([eye_landmarks[0].x, eye_landmarks[0].y, eye_landmarks[0].z]))
    return dist


def mouth_aspect_ratio(mouth_landmarks):
    if len(mouth_landmarks) < 309:
        return 0
    upper_lip_top = np.array([mouth_landmarks[13].x, mouth_landmarks[13].y, mouth_landmarks[13].z])
    lower_lip_bottom = np.array([mouth_landmarks[14].x, mouth_landmarks[14].y, mouth_landmarks[14].z])
    upper_lip_bottom = np.array([mouth_landmarks[12].x, mouth_landmarks[12].y, mouth_landmarks[12].z])
    lower_lip_top = np.array([mouth_landmarks[15].x, mouth_landmarks[15].y, mouth_landmarks[15].z])
    mouth_left_corner = np.array([mouth_landmarks[61].x, mouth_landmarks[61].y, mouth_landmarks[61].z])
    mouth_right_corner = np.array([mouth_landmarks[291].x, mouth_landmarks[291].y, mouth_landmarks[291].z])

    vertical_distance = np.linalg.norm(upper_lip_bottom - lower_lip_top)
    horizontal_distance = np.linalg.norm(mouth_left_corner - mouth_right_corner)
    mar = vertical_distance / horizontal_distance
    return mar


def detect_hand_shape(hand_landmarks):
    thumb_tip = -hand_landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP].y
    thumb_dip = -hand_landmarks[mp.solutions.hands.HandLandmark.THUMB_IP].y
    thumb_pip = -hand_landmarks[mp.solutions.hands.HandLandmark.THUMB_MCP].y
    index_finger_tip = -hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    index_finger_dip = -hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP].y
    index_finger_pip = -hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_finger_tip = -hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_finger_dip = -hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP].y
    middle_finger_pip = -hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_finger_tip = -hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y
    ring_finger_dip = -hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_DIP].y
    ring_finger_pip = -hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y
    pinky_tip = -hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP].y
    pinky_dip = -hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_DIP].y
    pinky_pip = -hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_PIP].y

    # Detect hand shape
    if (index_finger_tip > index_finger_dip 
        and index_finger_dip > index_finger_pip and middle_finger_tip > middle_finger_dip 
        and middle_finger_dip > middle_finger_pip and ring_finger_tip > ring_finger_dip 
        and ring_finger_dip > ring_finger_pip and pinky_tip > pinky_dip and pinky_dip > pinky_pip):
        return "Open Hand"
    elif (index_finger_tip < thumb_tip and middle_finger_tip < index_finger_tip
            and ring_finger_tip < middle_finger_tip and pinky_tip < ring_finger_tip):
        return "Closed Fist"
    elif index_finger_tip > thumb_tip and index_finger_tip > middle_finger_tip:
            return "Pointing"
    else:
        return "Unknown"

def detect_hand_position(hand_landmarks, face_landmarks):
    face_center = np.mean(
        [
            [face_landmarks[1].x, face_landmarks[1].y, face_landmarks[1].z],
            [face_landmarks[10].x, face_landmarks[10].y, face_landmarks[10].z],
            [face_landmarks[152].x, face_landmarks[152].y, face_landmarks[152].z],
            [face_landmarks[234].x, face_landmarks[234].y, face_landmarks[234].z],
            [face_landmarks[454].x, face_landmarks[454].y, face_landmarks[454].z],
        ], axis=0)

    wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
    distance = np.linalg.norm(
        np.array([wrist.x, wrist.y, wrist.z]) - face_center)

    # Adjusting threshold to detect proximity better
    if distance < 0.37:
        return "Near Face"
    else:
        return "Far from Face"

def main():
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    EYE_AR_THRESH = 0.2
    MOUTH_AR_THRESH = 0.2  # Lowered the threshold to detect smaller mouth openings

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands() as hands, mp_face_mesh.FaceMesh() as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in [149, 145]]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in [380, 385]]
                    
                    left_ear = eye_aspect_ratio(left_eye_landmarks)
                    right_ear = eye_aspect_ratio(right_eye_landmarks)
                    
                    if left_ear < EYE_AR_THRESH or right_ear < EYE_AR_THRESH:
                        cv2.putText(frame, "Eye blink detected", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    mouth_landmarks = face_landmarks.landmark
                    mar = mouth_aspect_ratio(mouth_landmarks)

                    if mar > MOUTH_AR_THRESH:
                        cv2.putText(frame, "Mouth open detected", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Detect Hand Shape
                        hand_shape = detect_hand_shape(hand_landmarks.landmark)
                        cv2.putText(frame, f"Hand Shape: {hand_shape}", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                        # Detect Hand Position Relative to Face
                        hand_position = detect_hand_position(hand_landmarks.landmark, face_landmarks.landmark)
                        cv2.putText(frame, f"Hand Position: {hand_position}", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

                        # Check if any finger touches the nose
                        nose_landmarks_indices = [1, 2, 3, 4, 5, 6, 195, 197, 5]
                        for index in nose_landmarks_indices:
                            nose = face_landmarks.landmark[index]
                            for finger_tip in [mp_hands.HandLandmark.THUMB_TIP,
                                               mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                               mp_hands.HandLandmark.RING_FINGER_TIP,
                                               mp_hands.HandLandmark.PINKY_TIP]:
                                finger_landmark = hand_landmarks.landmark[finger_tip]
                                distance = np.linalg.norm(np.array([nose.x, nose.y, nose.z]) - np.array([finger_landmark.x, finger_landmark.y, finger_landmark.z]))
                                if distance < 0.03:  # Adjust the threshold as needed
                                    cv2.putText(frame, "Hand touching nose", (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                    break

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
