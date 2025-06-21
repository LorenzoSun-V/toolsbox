# x86 rt 推理
import ctypes
import os
import numpy as np
import os
from inference.common import ImageData, DetBox, ImageInfo


class BaseYoloDetector:
    def __init__(self, library_path, weights_path, conf_threshold, nms_threshold, model_start_class_id,
                 init_func, detect_func, batch_func, destroy_func, use_rotation=False):  # <--
        self.lib = ctypes.CDLL(library_path)
        self.init_func = getattr(self.lib, init_func)
        self.detect_func = getattr(self.lib, detect_func)
        self.batch_func = getattr(self.lib, batch_func)
        self.destroy_func = getattr(self.lib, destroy_func)
        self.use_rotation = use_rotation  # <--

        self._setup_signatures()

        self.err_code = ctypes.c_int(0)
        self.instance = self.init_func(
            ctypes.c_char_p(weights_path.encode()),
            ctypes.byref(self.err_code),
            ctypes.c_float(conf_threshold),
            ctypes.c_float(nms_threshold),
            ctypes.c_int(model_start_class_id)
        )

        if self.err_code.value != 0:
            raise RuntimeError(f"模型初始化失败，错误码: {self.err_code.value}")
        print("模型初始化成功")

    def _setup_signatures(self):
        self.init_func.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
            ctypes.c_float, ctypes.c_float, ctypes.c_int
        ]
        self.init_func.restype = ctypes.c_void_p

        self.detect_func.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ImageData),
            ctypes.POINTER(DetBox), ctypes.POINTER(ctypes.c_int)
        ]
        self.detect_func.restype = ctypes.c_int

        self.batch_func.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ImageInfo), ctypes.c_int,
            ctypes.POINTER(DetBox)
        ]
        self.batch_func.restype = ctypes.c_int

        self.destroy_func.argtypes = [ctypes.c_void_p]
        self.destroy_func.restype = ctypes.c_int

    def detect(self, image):
        height, width, channels = image.shape
        image_data = ImageData(
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            width, height, channels
        )

        max_detections = 1000
        det_boxes = (DetBox * max_detections)()
        det_count = ctypes.c_int(0)

        err_code = self.detect_func(
            self.instance,
            ctypes.byref(image_data),
            det_boxes,
            ctypes.byref(det_count)
        )

        if err_code != 0:
            if err_code == 6:
                return []
            raise RuntimeError(f"检测失败，错误码: {err_code}")

        return [det_boxes[i] for i in range(det_count.value)]

    def batch_detect(self, images, rotations=None):
        image_count = len(images)
        if self.use_rotation:
            if rotations is None:
                rotations = [0] * image_count
            elif len(rotations) != image_count:
                raise ValueError("rotations长度必须和images相同")
        else:
            rotations = [0] * image_count  # <--

        img_datas = []
        image_infos = (ImageInfo * image_count)()
        for i, img in enumerate(images):
            if not (img.flags['C_CONTIGUOUS'] and img.dtype == np.uint8):
                img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w, c = img.shape
            img_datas.append(img.flatten())
            image_infos[i] = ImageInfo(w, h, c, rotations[i])

        all_image_data = np.concatenate(img_datas).astype(np.uint8)
        all_image_data_ptr = all_image_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        max_per_image = 100
        det_boxes = (DetBox * (max_per_image * image_count))()

        err_code = self.batch_func(
            self.instance,
            all_image_data_ptr,
            image_infos,
            ctypes.c_int(image_count),
            det_boxes
        )
        if err_code != 0:
            if err_code == 6:
                return []
            raise RuntimeError(f"批量检测失败，错误码: {err_code}")

        results_batch = []
        for i in range(image_count):
            base_idx = i * max_per_image
            dets = [det_boxes[base_idx + j] for j in range(max_per_image) if det_boxes[base_idx + j].classID != -1]
            results_batch.append(dets)

        return results_batch
    
    def close(self):
        if hasattr(self, 'instance') and self.instance is not None:
            err_code = self.destroy_func(self.instance)
            if err_code != 0:
                print(f"销毁模型实例失败，错误码: {err_code}")
            else:
                print("模型实例已成功销毁")
            self.instance = None


class Yolov8Detector(BaseYoloDetector):
    def __init__(self, library_path, weights_path, conf_threshold=0.25, nms_threshold=0.45, model_start_class_id=0):
        super().__init__(
            library_path, weights_path, conf_threshold, nms_threshold, model_start_class_id,
            init_func="InitYOLOE2EInferenceInstance",
            detect_func="InferenceYOLOE2EGetDetectResult",
            batch_func="BatchInferenceYOLOE2EGetDetectResult",
            destroy_func="DestoryYOLOE2EInferenceInstance",
            use_rotation=False  # <--
        )


class Yolov8obbDetector(BaseYoloDetector):
    def __init__(self, library_path, weights_path, conf_threshold=0.25, nms_threshold=0.45, model_start_class_id=100):
        super().__init__(
            library_path, weights_path, conf_threshold, nms_threshold, model_start_class_id,
            init_func="InitYolov8obbInferenceInstance",
            detect_func="InferenceYolov8obbGetDetectResult",
            batch_func="BatchInferenceYolov8obbGetDetectResult",
            destroy_func="DestoryYolov8obbInferenceInstance",
            use_rotation=True  # <--
        )
        

class Yolov8Pose(BaseYoloDetector):
    def __init__(self, library_path, weights_path, conf_threshold=0.25, nms_threshold=0.45, model_start_class_id=0):
        super().__init__(
            library_path, weights_path, conf_threshold, nms_threshold, model_start_class_id,
            init_func="InitYolov8obbInferenceInstance",
            detect_func="InferenceYolov8obbGetDetectResult",
            batch_func="BatchInferenceYolov8obbGetDetectResult",
            destroy_func="DestoryYolov8obbInferenceInstance",
            use_rotation=False  # <--
        )