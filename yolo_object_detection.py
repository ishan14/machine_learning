import cv2
import numpy as np
import glob
import random

read_net = cv2.dnn.readNet("C:/Users/ishan/OneDrive/Desktop/Machine_Learning/train_yolo_to_detect_custom_object/yolo_custom_detection/yolov3_training_last(1).weights", "C:/Users/ishan/OneDrive/Desktop/Machine_Learning/train_yolo_to_detect_custom_object/yolo_custom_detection/yolov3_testing.cfg")


classes = ["cone"]

# Images path
#images_path = glob.glob("C:/Users/ishan/OneDrive/Desktop/computer_vision/images/IGVC/*.jpg")
video_path = cv2.VideoCapture("C:/Users/ishan/OneDrive/Desktop/computer_vision/videos/IGCV2.mp4")


layer_names = read_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in read_net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
print(len(layer_names))
print(output_layers)
print(read_net.getUnconnectedOutLayers())
#random.shuffle(images_path)
# loop through all the images
#for img_path in images_path:
while True:
    ret,img = video_path.read()
    # Loading image
    #img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    read_net.setInput(blob)
    outs = read_net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font,1, color, 2)


    cv2.imshow("Image", img)
    #key = cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()