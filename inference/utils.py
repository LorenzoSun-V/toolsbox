import cv2
import math
import numpy as np


def draw_boxes(frame, results):
    color = (0,0,255)
    for i, obj in enumerate(results):
        class_id = obj.classID               
        x, y, w, h  = int(obj.x), int(obj.y), int(obj.w), int(obj.h) 
        label = f"{class_id} {obj.confidence:.2f}"
        if obj.radian == 0.0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        else:
            center = (x + w / 2.0, y + h / 2.0)
            size = (w, h)
            angle = radian_to_degree(obj.radian)
            rotated_rect = (center, size, angle)
            box = cv2.boxPoints(rotated_rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


# 弧度转角度函数
def radian_to_degree(radian):
    return radian * 180.0 / math.pi