import cv2
import dlib
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import pandas as pd
from datetime import datetime
import time

# Initialize dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("a")

# Initialize MTCNN face detector and InceptionResnetV1 model
mtcnn = MTCNN(keep_all=True)  # MTCNN for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # InceptionResnetV1 for face feature extraction

# Load the known images and encode them
known_names_file = "known_names.txt"
with open(known_names_file, "r") as file:
    known_face_names = [line.strip() for line in file.readlines()]  # Names of the people in the database
known_face_encodings = []
print("b")

for name in known_face_names:
    image = cv2.imread(f"{name}.jpg")  # Assuming images are named as "Alice.jpg", "Bob.jpg", etc.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face using dlib
    rects = detector(rgb_image, 1)
    print("c")

    if len(rects) > 0:
        # Assume there's only one face in each image
        rect = rects[0]

        # Use dlib shape predictor to get facial landmarks
        landmarks = predictor(rgb_image, rect)

        # Align face using dlib
        aligned_face = dlib.get_face_chip(rgb_image, landmarks, size=96)
        print("d")

        if aligned_face is not None:
            # Preprocess face for facenet-pytorch
            aligned_face = cv2.resize(aligned_face, (160, 160))  # Resize to match facenet-pytorch input size
            aligned_face = np.transpose(aligned_face, (2, 0, 1))  # Transpose to (C, H, W)
            aligned_face = aligned_face / 255.0  # Normalize to [0, 1]
            aligned_face_tensor = torch.tensor(aligned_face, dtype=torch.float32).unsqueeze(0)

            # Get face embedding using InceptionResnetV1
            encoding = resnet(aligned_face_tensor).detach().numpy()
            known_face_encodings.append(encoding)
            print("e")

# Create an empty DataFrame to store matched names, dates, and times
df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Checkout Time'])

# Capture video from the first camera
video_capture1 = cv2.VideoCapture(0)

# Initialize the second camera
video_capture2 = None
try:
    # Try to capture video from the second camera
    video_capture2 = cv2.VideoCapture(2)
except Exception as e:
    # Handle the exception if the second camera is not available
    print("Error: Second camera not available. Exception:", e)
print("f")
# Threshold for considering a match
threshold = 0.85

# Set the delay (in seconds)
delay_seconds = 5  # Adjust this value as needed

# Initialize a variable to keep track of the last write time
last_write_time = time.time()

while True:
    # Capture frame-by-frame from the first camera
    ret1, frame1 = video_capture1.read()

    # Capture frame-by-frame from the second camera (if available)
    ret2, frame2 = video_capture2.read() if video_capture2 is not None else (None, None)

    # Convert the frame from BGR color to RGB color for both cameras
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) if ret1 else None
    rgb_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) if ret2 else None

    # Detect faces using MTCNN for both cameras
    boxes1, _ = mtcnn.detect(rgb_frame1) if rgb_frame1 is not None else (None, None)
    boxes2, _ = mtcnn.detect(rgb_frame2) if rgb_frame2 is not None else (None, None)
    print("g")

    # Loop through each detected face in the first camera
    if boxes1 is not None:
        for box in boxes1:
            # Convert box to dlib format
            rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))

            # Use dlib shape predictor to get facial landmarks
            landmarks = predictor(rgb_frame1, rect)
            print("h")

            # Align face using dlib
            aligned_face = dlib.get_face_chip(rgb_frame1, landmarks, size=96)

            if aligned_face is not None:
                # Preprocess face for facenet-pytorch
                aligned_face = cv2.resize(aligned_face, (160, 160))  # Resize to match facenet-pytorch input size
                aligned_face = np.transpose(aligned_face, (2, 0, 1))  # Transpose to (C, H, W)
                aligned_face = aligned_face / 255.0  # Normalize to [0, 1]

                # Get face embedding using InceptionResnetV1
                encoding = resnet(torch.tensor(aligned_face, dtype=torch.float32).unsqueeze(0)).detach().numpy()

                # Compare the face with known faces
                distances = [np.linalg.norm(encoding - known_encoding) for known_encoding in known_face_encodings]

                # Check if the minimum distance is less than the threshold
                if min(distances) < threshold:
                    # If a match is found, add the name, date, and time to the DataFrame
                    name = known_face_names[distances.index(min(distances))]
                else:
                    # If no match is found, assign the name as "Unknown"
                    name = "Unknown"

                timestamp = datetime.now()
                df = df.append({'Name': name, 'Date': timestamp.date(), 'Time': timestamp.time()}, ignore_index=True)

                # Draw a rectangle around the face and display the name for the first camera
                cv2.rectangle(frame1, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
                cv2.putText(frame1, name, (rect.left() + 6, rect.bottom() - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

    # Loop through each detected face in the second camera
    if boxes2 is not None:
        for box in boxes2:
            # Convert box to dlib format
            rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))

            # Use dlib shape predictor to get facial landmarks
            landmarks = predictor(rgb_frame2, rect)

            # Align face using dlib
            aligned_face = dlib.get_face_chip(rgb_frame2, landmarks, size=96)

            if aligned_face is not None:
                # Preprocess face for facenet-pytorch
                aligned_face = cv2.resize(aligned_face, (160, 160))  # Resize to match facenet-pytorch input size
                aligned_face = np.transpose(aligned_face, (2, 0, 1))  # Transpose to (C, H, W)
                aligned_face = aligned_face / 255.0  # Normalize to [0, 1]

                # Get face embedding using InceptionResnetV1
                encoding = resnet(torch.tensor(aligned_face, dtype=torch.float32).unsqueeze(0)).detach().numpy()

                # Compare the face with known faces
                distances = [np.linalg.norm(encoding - known_encoding) for known_encoding in known_face_encodings]

                # Check if the minimum distance is less than the threshold
                if min(distances) < threshold:
                    # If a match is found, add the name, date, and time to the DataFrame
                    name = known_face_names[distances.index(min(distances))]
                else:
                    # If no match is found, assign the name as "Unknown"
                    name = "Unknown"

                timestamp = datetime.now()
                df = df.append({'Name': name, 'Date': timestamp.date(), 'Time': timestamp.time()}, ignore_index=True)

                # Draw a rectangle around the face and display the name for the second camera
                cv2.rectangle(frame2, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
                cv2.putText(frame2, name, (rect.left() + 6, rect.bottom() - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

    # Display the resulting images from both cameras (if available)
    if ret1:
        cv2.imshow('Camera 1', frame1)
    if ret2:
        cv2.imshow('Camera 2', frame2)

    # Check if it's time to write to the Excel file
    if time.time() - last_write_time >= delay_seconds:
        # Save the DataFrame to an Excel file
        df.to_excel('matched_names.xlsx', index=False)
        # Update the last write time
        last_write_time = time.time()

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close the OpenCV windows
video_capture1.release()
if video_capture2 is not None:
    video_capture2.release()
cv2.destroyAllWindows()

# Read the Excel file into a DataFrame
df = pd.read_excel('matched_names.xlsx')

# Drop duplicates based on the 'Name' column, keeping the first occurrence
df = df.drop_duplicates(subset=['Name'], keep='first')

# Save the updated DataFrame back to the Excel file
df.to_excel('matched_names.xlsx', index=False)
