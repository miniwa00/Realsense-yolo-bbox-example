import cv2
from realsense_depth import *
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")
font = cv2.FONT_HERSHEY_SIMPLEX


def cv2_show(color: tuple):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img, (x1, y2), (x1 + 80, y2 - 50), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, class_name, (x1, y1 - 10), font, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        "{}%".format(round(conf * 100, 2)),
        (x1 + 10, y2 - 30),
        font,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "{}cm".format(distance / 10),
        (x1 + 10, y2 - 10),
        font,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


cam = Realsense()
cv2.namedWindow("Color frame")
while True:
    _, depth_frame, img = cam.get_frame()
    try:
        res = model(img)
        results = res[0].boxes
        for boxes in results:
            x1, y1, x2, y2 = boxes.xyxy[0].numpy().astype(int)
            class_name = res[0].names[int(boxes.cls[0])]
            if class_name != "person":
                continue
            conf = boxes.conf[0].numpy()
            if conf < 0.6:
                continue
            point = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = depth_frame[point[1], point[0]]
            if (distance / 10) >= 100:
                cv2_show((0, 0, 255))
            else:
                cv2_show((0, 255, 0))
    except:
        pass
    cv2.imshow("Color frame", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
