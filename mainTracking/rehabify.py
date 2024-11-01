import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import asyncio
import time

# Set up hand model and drawing utilities
mp_drawing = mp.solutions.drawing_utils  # Corrected to drawing_utils
mp_hands = mp.solutions.hands  # Hand model

# Create a touch threshold for how far the finger can be considered as touching
touchThreshold = 0.05  # Threshold value

def calDist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Dictionary to map finger touches to specific key presses
fingerToTouches = {
    'IndexFinger': 'a',
    'MiddleFinger': 'w',
    'RingFinger': 's',
    'PinkyFinger': 'd'
}

# Dictionary to track which keys are currently being held down
held_keys = {}

async def press_key_async(key):
    if not held_keys.get(key):
        pyautogui.keyDown(key)
        await asyncio.sleep(0.01)
        held_keys[key] = True  # Mark this key as held down

async def release_key_async(key):
    if held_keys.get(key):
        pyautogui.keyUp(key)
        await asyncio.sleep(0.01)
        held_keys[key] = False  # Mark this key as not held down

async def main():
    cap = cv2.VideoCapture(0)  # Open webcam
   


    # Initialize Mediapipe Hands model
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break if frame is not read correctly
             # Resize frame
            frame = cv2.resize(frame, (640, 480))  # Adjust size as needed
            # Convert the frame to RGB for Mediapipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render joints on the detected hand
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                    )

                    # Extract landmarks and calculate distances
                    landMarks = hand.landmark
                    thumbTip = (landMarks[4].x, landMarks[4].y)
                    touches = {}
                    fingertips = [8, 12, 16, 20]
                    fingerNames = ['IndexFinger', 'MiddleFinger', 'RingFinger', 'PinkyFinger']

                    for i, tip in enumerate(fingertips):
                        finger_tip = (landMarks[tip].x, landMarks[tip].y)
                        dist = calDist(thumbTip, finger_tip)
                        if dist < touchThreshold:
                            touches[fingerNames[i]] = True
                        else:
                            touches[fingerNames[i]] = False

                    # Check for touches and trigger key presses
                    for finger, is_touching in touches.items():
                        keyToPress = fingerToTouches.get(finger)
                        if keyToPress:
                            if is_touching:
                                await press_key_async(keyToPress)  # Non-blocking key press
                            else:
                                await release_key_async(keyToPress)  # Non-blocking key release

            # Display the frame with drawn landmarks
            cv2.imshow('Rehabify Prototype', image)

            # Quit the program when 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function using asyncio
asyncio.run(main())
