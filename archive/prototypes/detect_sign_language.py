import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import json

# 1. SETUP MEDIAPIPE
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 2. LOAD YOUR KERAS MODEL
# Note: Use the exact filename you uploaded: 'signlanguauge_model.h5'
model = load_model('signlanguauge_model.h5') 

# 3. LOAD LABELS
# We need to map the predicted number (e.g., 0) to a word (e.g., "Hello")
# If you don't have label_encoder.npy, we use your label_map.json
with open('label_map.json', 'r') as f:
    label_map = json.load(f)
# Create a reverse map: {0: "all", 1: "bed", ...}
idx_to_label = {v: k for k, v in label_map.items()}

def extract_keypoints(results):
    """
    Extracts and flattens keypoints to match the CSV format from training.
    CRITICAL: This order must match exactly how your 'keypoint.csv' was created.
    """
    # 1. Face (468 points * 3 coords = 1404) - adjust if your training didn't use face
    # 2. Pose (33 points * 4 coords = 132) 
    # 3. Left Hand (21 points * 3 coords = 63)
    # 4. Right Hand (21 points * 3 coords = 63)
    
    # Example for standard holistic extraction (Sequence MUST match training):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # If your model expects face data, uncomment this:
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # return np.concatenate([pose, face, lh, rh])
    
    # If your model is trained ONLY on Pose + Hands (Common for simple signs):
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Process Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks (Visual Debugging)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        try:
            # Extract Keypoints
            keypoints = extract_keypoints(results)
            
            # Reshape for Model (1 sample, N features)
            input_data = np.expand_dims(keypoints, axis=0)
            
            # Predict
            prediction = model.predict(input_data)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Only show if confidence is high enough
            if confidence > 0.7:
                word = idx_to_label.get(class_id, "Unknown")
                
                # Display on Screen
                cv2.putText(image, f'{word} ({confidence:.2f})', (10, 50), 
                           cv2.putText.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                 cv2.putText(image, f'Low Confidence: {confidence:.2f}', (10, 50), 
                           cv2.putText.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        except Exception as e:
            # likely input shape mismatch
            pass

        cv2.imshow('Sign Language Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()