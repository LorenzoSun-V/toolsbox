import ctypes


class ImageData(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_ubyte)),  # 图像数据指针
        ("width", ctypes.c_int),                    # 图像宽度
        ("height", ctypes.c_int),                   # 图像高度
        ("channels", ctypes.c_int)                 # 图像通道数
    ]


class DetBox(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),                     # 左上角x坐标
        ("y", ctypes.c_float),                     # 左上角y坐标
        ("h", ctypes.c_float), 
        ("w", ctypes.c_float),   
        ("confidence", ctypes.c_float),
        ("classID", ctypes.c_int),                   
        ("radian", ctypes.c_float),
        ("keypoints",  ctypes.c_float * (17*3)),           
    ]


class ImageInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),                    # 图像宽度
        ("height", ctypes.c_int),                   # 图像高度
        ("channels", ctypes.c_int),                   # 图像步长
        ("isRotate", ctypes.c_int)
    ]