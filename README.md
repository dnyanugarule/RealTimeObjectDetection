# Real Time Object Detection

**Introduction**

YOLO (You Only Look Once) is a method/way to do object detection. It is the algorithm/strategy behind how the code is going to detect objects in the image. The official implementation of this idea is available through DarkNet (neural net implementation from the ground up in C from the author). It is available on github for people to use.

Earlier detection frameworks, looked at different parts of the image multiple times at different scales and repurposed image classification technique to detect objects. This approach is slow and inefficient. YOLO takes entirely different approach. It looks at the entire image only once and goes through the network once and detects objects. Hence the name. It is very fast. That’s the reason it has got so popular. There are other popular object detection frameworks like Faster R-CNN and SSD that are also widely used.

**Dependencies**

opencv
numpy
YOLO v8

**System Design**

Here is a the system design for YOLO object detection using python and OpenCV-

**Data Collection and Preparation** - Firstly, collected a large dataset of images and videos with the objects you want to detect. This dataset needs to be varied and typical of the scenarios in real life where you want to find objects. Use bounding boxes or masks to describe the items in each frame of an image or video. The YOLO model will be trained using this labelled dataset.
**YOLO Model Architecture **- The YOLO model architecture consists of deep convolutional neural network that can process entire images and output bounding boxes and class probabilities for the detected objects. There are several variations of YOLO, including YOLOv1, YOLOv2, YOLOv3, and YOLOv4.
**Training the Model** - Using a loss function that penalises inaccurate item detections and localization mistakes, train the YOLO model on the labelled dataset. Use a powerful GPU to quicken the training process.
Optimization - After training, optimise the YOLO model by reducing its size and increasing its speed without sacrificing performance. This can be done using techniques like model quantization, pruning, and knowledge distillation.

**Applications of the project**
Object detection using CNN has many practical applications in various fields, including:

**# Self-driving cars:** Object detection is used to identify and track objects such as pedestrians, cars, and traffic signs on the road.
**Surveillance:** Object detection can be used for monitoring and detecting objects and individuals in surveillance cameras and security systems.
**Robotics:** Object detection is used to identify and locate objects in a robot's environment, allowing it to navigate and interact with its surroundings.
Medical imaging: Object detection is used in medical imaging to identify and locate abnormalities such as tumours, lesions, or other medical conditions.
**Agriculture:** Object detection can be used to monitor and identify crops, weeds, and pests, enabling precision farming techniques.

**OUTPUT**

![```![python](https://)
alt text

```](object-detection-1.jpg)

```
