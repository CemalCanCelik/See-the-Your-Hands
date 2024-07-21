# Hand Detection and Tracking with OpenCV and MediaPipe

This project demonstrates real-time hand detection and tracking using OpenCV and MediaPipe. The application captures video from the default camera, processes each frame to detect hands, and displays the results with annotations indicating whether the detected hand is the left or right hand.

## Key Features:
- **Real-time Processing**: The video stream is processed in real-time for hand detection and tracking.
- **Hand Detection**: Detects up to two hands in the video frame.
- **Hand Classification**: Classifies the detected hands as left or right.
- **Annotations**: Displays annotations on the video frame indicating whether the detected hand is the left or right hand.

## Requirements:
- OpenCV
- MediaPipe
- Google Protobuf

## How to Run:
1. Ensure you have the required libraries installed.
2. Run the script using a Python environment with access to a camera.

## Code Description:
```python
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

# Initialize MediaPipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,         # Process the video stream in real-time
    model_complexity=1,              # Set the complexity of the hand detection model
    min_detection_confidence=0.75,   # Minimum confidence for the hand detection to be considered successful
    min_tracking_confidence=0.75,    # Minimum confidence for the hand tracking to be considered successful
    max_num_hands=2                  # Maximum number of hands to detect
)

# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()            # Read a frame from the video capture
    img = cv2.flip(img, 1)               # Flip the image horizontally
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    results = hands.process(imgRGB)      # Process the frame to detect hands
    
    if results.multi_hand_landmarks:
        # Check if both hands are detected
        if len(results.multi_handedness) == 2:
            cv2.putText(img, 'Both Hands', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        else:
            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
                if label == 'Left':
                    cv2.putText(img, label + ' Hand', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                if label == 'Right':
                    cv2.putText(img, label + ' Hand', (460, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Image', img)             # Display the frame with annotations
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
