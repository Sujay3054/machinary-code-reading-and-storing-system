import cv2
from paddleocr import PaddleOCR
import numpy as np
from db import MySQLClient

# Initialize PaddleOCR (English only, with angle classification for rotated text)
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Use GPU if available

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

    # Resize for better OCR accuracy (optional)
    frame_resized = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Perform OCR with PaddleOCR
    results = ocr.ocr(frame_resized, cls=True)

    # Process results
    for line in results[0] if results and results[0] else []:
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]

        # Convert resized bbox coordinates back to original frame
        top_left = tuple(map(int, (bbox[0][0] / 1.5, bbox[0][1] / 1.5)))
        bottom_right = tuple(map(int, (bbox[2][0] / 1.5, bbox[2][1] / 1.5)))

        # Draw bounding box and text
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save result in the database
        mysql_client.insert_result(bbox, text, confidence)

    cv2.imshow("Live Feed with OCR", frame)
    # Check for ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        print("Terminating...")
        break
# Release resources
cap.release()
cv2.destroyAllWindows()
mysql_client.close()