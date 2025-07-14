try:
    import sys
    sys.path.append('libs')
    import py_yoloe2e
    import py_yolov8obb
except ImportError as e:
    print(f"错误: 无法导入所需的Python绑定: {e}")
    print("请确保 py_videodec, py_yoloe2e, py_yolov8obb 模块已正确编译并位于PYTHONPATH中。")
    exit(1)


class YOLODetector:
    def __init__(self, model_path, model_type, class_names, device_id=0, conf_threshold=0.25, nms_threshold=0.45, model_start_class_id=0):
        self.model_path = model_path
        self.model_type = model_type
        self.class_names = class_names
        self.device_id = device_id
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_start_class_id = model_start_class_id
        self.detector = self._create_detector()
    
    def _create_detector(self):
        if self.model_type == 'yoloe2e':
            return py_yoloe2e.PyYOLOE2EDetector(
                model_path=self.model_path,
                device_id=self.device_id,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                model_start_class_id=self.model_start_class_id
            )
        elif self.model_type == 'yolov8obb':
            return py_yolov8obb.Yolov8obbDetector(
                model_path=self.model_path,
                device_id=self.device_id,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                model_start_class_id=self.model_start_class_id
            )
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
    
    def detect(self, image):
        if not hasattr(self.detector, 'detect'):
            raise RuntimeError("检测器未正确初始化或不支持 detect 方法")
        return self.detector.detect(image)

# a = YOLODetector(
#     model_path='/data/lorenzo/nofar/Detection/model_version/model8e2e_b32m_20250625_person_attribute_v0.1.4.bin',
#     model_type='yoloe2e',
#     class_names=["nomask", "smoking", "nowf", "fall", "sleep", "tx", "wsf"],
#     device_id=0,
#     conf_threshold=0.25,
#     nms_threshold=0.45,
#     model_start_class_id=0
# )
# import cv2
# from inference.utils import draw_alltype
# image = cv2.imread('/home/lorenzo/Code/Detection/ultralytics/ultralytics/assets/bus.jpg')
# results = a.detect(image)
# print(f"检测到目标数量: {len(results)}")
# image_draw = draw_alltype(a.class_names, image, results)
# # print(results)
# cv2.imwrite('result.jpg', image_draw)