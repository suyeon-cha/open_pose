from dis import dis
from pydoc import classname
import cv2
import pyrealsense2
from realsense_depth import *

# Opencv DNN
net = cv2.dnn.readNet("/home/suyeoncha/catkin_ws/src/dnn_model/yolov4-tiny.weights", "/home/suyeoncha/catkin_ws/src/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("/home/suyeoncha/catkin_ws/src/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
        
print("Objects list")
print(classes)


# Initialze Camera Intel Realsense
dc = DepthCamera()

while True:

    # Get frames
    ret, depth_frame, color_frame = dc.get_frame()

    # Object detection
    (class_ids, scores, bboxes) = model.detect(color_frame)
    for (class_id, score, bbox) in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        # Set point to the center of bbox
        point = ((int)((x+w)/2), (int)((y+h)/2))
        distance = depth_frame[point[1], point[0]]

        class_name = classes[class_id]
        
        cv2.putText(color_frame, "{}, {}mm".format(class_name, distance), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(color_frame, (x, y), (x+w, y+h), (200, 0, 50), 3)

    print("class ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)

    # Show distance for a specific point
    
    # cv2.circle(color_frame, point, 4, (0, 0, 255))
    # distance = depth_frame[point[1], point[0]]

    # cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    # cv2.imshow("Depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)