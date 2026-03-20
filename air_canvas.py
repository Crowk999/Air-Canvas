import cv2
import numpy as np
import mediapipe as mp
from collections import deque

 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Drawing points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
kpoints = [deque(maxlen=1024)]

blue_index = green_index = red_index = black_index = 0

# Default color
colorIndex = 0

# Canvas
paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255

#  Gesture Detection
hand_buffer = []

def is_hand_open(hand_landmarks):
    lm = hand_landmarks.landmark

    fingers = []

    # Index, Middle, Ring, Pinky
    fingers.append(lm[8].y < lm[6].y)
    fingers.append(lm[12].y < lm[10].y)
    fingers.append(lm[16].y < lm[14].y)
    fingers.append(lm[20].y < lm[18].y)

    # Thumb (basic check)
    fingers.append(lm[4].x < lm[3].x)

    return fingers.count(True) >= 4


def stable_hand_open(hand_landmarks):
    global hand_buffer

    current = is_hand_open(hand_landmarks)
    hand_buffer.append(current)

    if len(hand_buffer) > 5:
        hand_buffer.pop(0)

    return hand_buffer.count(True) >= 4

# UI Buttons 
def draw_buttons(frame):
    cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    cv2.putText(frame, "CLEAR", (49,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    cv2.putText(frame, "BLUE", (175,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
    cv2.putText(frame, "GREEN", (285,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
    cv2.putText(frame, "RED", (405,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.rectangle(frame, (505,1), (600,65), (0,0,0), 2)
    cv2.putText(frame, "BLACK", (510,33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

#  Webcam 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    draw_buttons(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            h, w, _ = frame.shape

            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

            #  Button Interaction 
            if y <= 65:
                if 40 <= x <= 140:
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    kpoints = [deque(maxlen=1024)]

                    blue_index = green_index = red_index = black_index = 0
                    paintWindow[67:, :, :] = 255

                elif 160 <= x <= 255:
                    colorIndex = 0
                elif 275 <= x <= 370:
                    colorIndex = 1
                elif 390 <= x <= 485:
                    colorIndex = 2
                elif 505 <= x <= 600:
                    colorIndex = 3

            #  Drawing Logic 
            else:
                if stable_hand_open(hand_landmarks):
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft((x, y))
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft((x, y))
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft((x, y))
                    elif colorIndex == 3:
                        kpoints[black_index].appendleft((x, y))
                else:
                    # Stop drawing → new stroke
                    bpoints.append(deque(maxlen=1024))
                    gpoints.append(deque(maxlen=1024))
                    rpoints.append(deque(maxlen=1024))
                    kpoints.append(deque(maxlen=1024))

                    blue_index += 1
                    green_index += 1
                    red_index += 1
                    black_index += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw Lines 
    points = [bpoints, gpoints, rpoints, kpoints]
    colors = [(255,0,0), (0,255,0), (0,0,255), (0,0,0)]

    for i in range(4):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k], colors[i], 5)
                cv2.line(paintWindow, points[i][j][k-1], points[i][j][k], colors[i], 5)

    # Display
    cv2.imshow("Air Canvas", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()