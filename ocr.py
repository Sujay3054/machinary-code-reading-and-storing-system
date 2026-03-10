import cv2
from paddleocr import PaddleOCR
import numpy as np
from db import MySQLClient

# Initialize PaddleOCR (English only, with angle classification for rotated text)
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # GPU enabled by default if available

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MySQL client to interact with the database
mysql_client = MySQLClient()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera.")
        break

    # Preprocess: Resize for better detection (optional)
    frame_resized = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Perform OCR with PaddleOCR
    results = ocr.ocr(frame_resized, cls=True)

    # Draw bounding boxes and text on original frame
    for line in results[0] if results and results[0] else []:
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]
        # Convert bbox coordinates to original frame size (undo resize)
        top_left = tuple(map(int, (bbox[0][0] / 1.5, bbox[0][1] / 1.5)))
        bottom_right = tuple(map(int, (bbox[2][0] / 1.5, bbox[2][1] / 1.5)))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Insert result into the database
        mysql_client.insert_result(bbox, text, confidence)

    # Display the live feed with detected text
    cv2.imshow("Live Feed with OCR", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' key to terminate the process
        print("Terminating process...")
        break

cap.release()
cv2.destroyAllWindows()

# Close the database connection after the process
mysql_client.close()

