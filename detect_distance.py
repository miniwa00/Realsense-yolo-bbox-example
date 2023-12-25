import cv2
from realsense_depth import *
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

point = (400, 300)
font = cv2.FONT_HERSHEY_SIMPLEX
names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def cv2_show(color: tuple):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, class_name, (x1, y1 - 10), font, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        conf,
        (x1 + 200, y1 - 10),
        font,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "{}cm".format(distance / 10),
        (x1 + 100, y1 - 10),
        font,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


cam = Realsense()
cv2.namedWindow("Color frame")
while True:
    _, depth_frame, img = cam.get_frame()
    try:
        results = model(img)
        box_list = results[0].boxes
        for box in box_list:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            class_name = names[int(box.cls[0].numpy())]
            if class_name != "person":
                continue
            conf = str(box.conf[0].numpy())
            if float(conf) < 0.6:
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
