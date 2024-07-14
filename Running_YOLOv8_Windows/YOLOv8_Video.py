
# from ultralytics import YOLO
# import cv2
# import math
# cap=cv2.VideoCapture('../Videos/car.mp4')

# frame_width=int(cap.get(3))
# frame_height = int(cap.get(4))

# out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# model=YOLO("../YOLO-Weights/yolov8n.pt")
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
# while True:
#     success, img = cap.read()
#     # Doing detections using YOLOv8 frame by frame
#     #stream = True will use the generator and it is more efficient than normal
#     results=model(img,stream=True)
#     #Once we have the results we can check for individual bounding boxes and see how well it performs
#     # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
#     # we will loop through each of the bouning box
#     for r in results:
#         boxes=r.boxes
#         for box in boxes:
#             x1,y1,x2,y2=box.xyxy[0]
#             #print(x1, y1, x2, y2)
#             x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
#             print(x1,y1,x2,y2)
#             cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
#             #print(box.conf[0])
#             conf=math.ceil((box.conf[0]*100))/100
#             cls=int(box.cls[0])
#             class_name=classNames[cls]
#             label=f'{class_name}{conf}'
#             t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#             #print(t_size)
#             c2 = x1 + t_size[0], y1 - t_size[1] - 3
#             cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
#             cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
#     out.write(img)
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF==ord('1'):
#         break
# out.release()





from ultralytics import YOLO
import cv2
import math

# Initialize video capture
cap = cv2.VideoCapture('../Videos/cars2.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

# Load the YOLO model
model = YOLO("../YOLO-Weights/yolov8n.pt")

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Loop through video frames
while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame.")
        break

    # Perform YOLO prediction
    results = model(img, stream=True)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Write the frame with detections to the output video
    out.write(img)
    
    # Display the frame
   # cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
