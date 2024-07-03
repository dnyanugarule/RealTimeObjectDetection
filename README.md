
YOLO-object-detection-with-OpenCV



Object detection using YOLO object detector

Detect objects in both images and video streams using Deep Learning, OpenCV, and Python.
I’ll be using YOLOv3 in this project, in particular, YOLO trained on the COCO dataset.

The COCO dataset consists of 80 labels, including, but not limited to:

People
Bicycles
Cars and trucks
Airplanes
Stop signs and fire hydrants
Animals, including cats, dogs, birds, horses, cows, and sheep, to name a few
Kitchen and dining objects, such as wine glasses, cups, forks, knives, spoons, etc. …and much more!




## Installation



```bash
  
   
   pip install numpy
pip install opencv-python
```


YOLO (You Only Look Once)


Download the pre-trained YOLO v3 weights file from this link and place it in the current directory or you can directly download to the current directory in terminal using

$ wget https://pjreddie.com/media/files/yolov3.weights

Provided all the files are in the current directory, below command will apply object detection on the input image dog.jpg.

$ python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt



## Screenshots

![OUTPUT](C:\Users\dnyan\Downloads\object-detection.jpg)

