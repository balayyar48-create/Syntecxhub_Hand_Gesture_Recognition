import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def detect_gesture(landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if landmarks[tips[0]].x < landmarks[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[tips[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [0, 1, 1, 1, 1]:
        return "OPEN PALM"
    elif fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP"
    else:
        return "UNKNOWN"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks.landmark)

    cv2.putText(frame, gesture, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
