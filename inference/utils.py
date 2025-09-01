import cv2
import math
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import json


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


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def get_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)


def draw_txt(frame, x1, y1, label):
    cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype('inference/simfang.ttf', 22, encoding="utf-8")
    draw.text((x1, max(y1 - 20,0)), f"{label}", (255, 0, 0), font=font)
    frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return frame


# def draw_alltype(class_names, frame, results):    
#     for obj in results:
#         class_name = class_names[obj.classID]
#         color = get_color(obj.classID)           
#         x, y, w, h  = int(obj.x), int(obj.y), int(obj.w), int(obj.h) 
#         label = f"{class_name} {obj.confidence:.2f}"
#         if obj.radian == 0.0:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#         else:
#             center = (x + w / 2.0, y + h / 2.0)
#             size = (w, h)
#             angle = radian_to_degree(obj.radian)
#             rotated_rect = (center, size, angle)
#             box = cv2.boxPoints(rotated_rect)
#             box = np.int0(box)
#             cv2.drawContours(frame, [box], 0, color, 2)
#         frame = draw_txt(frame,x,y,label)
#     return frame

def draw_alltype(class_names, frame, results):
    try:      
        for obj in results:
            if len(obj)==7:
                x, y, w, h, confidence, class_id,radian = obj
            else:
                x, y, w, h, confidence, class_id = obj
                radian = 0.0
            x, y, w, h = int(x), int(y), int(w), int(h)
            # x, y, w, h,confidence, class_id ,radian  = int(obj.x), int(obj.y), int(obj.w), int(obj.h) ,obj.confidence,obj.classID,obj.radian
            class_name = class_names[class_id]
            # print(class_name)
            color = get_color(class_id)           
            label = f"{class_name} {confidence:.2f}"
            if class_id == 5:
                y -= 20
            if radian == 0.0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            else:
                center = (x + w / 2.0, y + h / 2.0)
                size = (w, h)
                angle = radian_to_degree(radian)
                rotated_rect = (center, size, angle)
                box = cv2.boxPoints(rotated_rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, color, 2)
            frame = draw_txt(frame,x,y,label)
        return frame
    except Exception as e:
        print(f"draw_alltype Error: {e}")
        return frame  # 保守返回原图

# 弧度转角度函数
def radian_to_degree(radian):
    return radian * 180.0 / math.pi


def detbox_to_shape_rectangle(obj, class_names):
    # 转为 rectangle 4点顺时针
    x1, y1, w, h, score, class_id = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4]), int(obj[5])
    points = [
        [x1, y1],
        [x1 + w, y1],
        [x1 + w, y1 + h],
        [x1, y1 + h]
    ]
    label = class_names[class_id] if class_id < len(class_names) else str(class_id)
    return {
        "kie_linking": [],
        "label": label,
        "score": score,
        "points": points,
        "group_id": None,
        "description": None,
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {}
    }


def detbox_to_shape_rotation(obj, class_names):
    """
    将您的特定 OBB (旋转框) 检测结果转换为 Anylabeling 的 'rotation' 格式。
    obj: [x_top_left, y_top_left, w, h, confidence, class_id, radian]
    """
    x_tl, y_tl, w, h, score, class_id, radian = obj
    x_tl, y_tl, w, h, score, class_id, radian = float(x_tl), float(y_tl), float(w), float(h), float(score), int(class_id), float(radian)

    # 根据左上角点、宽高和旋转角度计算四个顶点
    center_x = x_tl + w / 2
    center_y = y_tl + h / 2
    
    cos_r = math.cos(radian)
    sin_r = math.sin(radian)

    corners = np.array([
            [-w/2, -h/2],
            [w/2, -h/2], 
            [w/2, h/2],
            [-w/2, h/2]
        ])

    rotated_corners_np = np.array([
            [cos_r * corner[0] - sin_r * corner[1] + center_x,
             sin_r * corner[0] + cos_r * corner[1] + center_y]
            for corner in corners
        ], dtype=np.float32)
    
    points = rotated_corners_np.tolist()

    label = class_names[class_id] if class_id < len(class_names) else str(class_id)
    
    return {
        "kie_linking": [],
        "label": label,
        "score": score,
        "points": points,
        "group_id": None,
        "description": None,
        "difficult": False,
        "shape_type": "rotation", # 根据您的JSON示例，使用 'rotation'
        "flags": {},
        "attributes": {},
        "direction": radian # 您可以选在在此处保留原始角度信息
    }


def save_anylabeling_json(img_path, shapes, out_dir, img_shape):
    json_data = {
        "version": "3.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": img_shape[0],
        "imageWidth": img_shape[1],
        "description": ""
    }
    base_name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    json_path = os.path.join(out_dir, base_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)