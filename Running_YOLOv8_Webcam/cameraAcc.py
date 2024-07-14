import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('test_frame.jpg', frame)
        print("Frame captured and saved successfully.")
    cap.release()
