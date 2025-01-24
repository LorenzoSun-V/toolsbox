import cv2
import os
import os.path as osp
import numpy as np
import json
from glob import glob
from tqdm import tqdm


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.array): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    

class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
    """

    def __init__(self, im, line_width=None):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        self.im = im
        self.tf = max(self.lw - 1, 1)  # font thickness
        self.sf = self.lw / 3  # font scale

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label."""
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.sf,
                        txt_color,
                        thickness=self.tf,
                        lineType=cv2.LINE_AA)

label_map = {
    "joint": 0,
    "brokenlen": 1,
    "embedding": 2,
    "fracture": 3,
    "brokenwire": 4,
    "twitch": 5,
    "missing": 6
}

def vis_json(folder_path):
    colors = Colors()
    img_dirs = sorted(glob(osp.join(folder_path, "*.jpg")))
    dst_dir = osp.join(folder_path, "vis")
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    for img_dir in tqdm(img_dirs):
        # print(img_dir)
        json_dir = img_dir.replace(".jpg", ".json")
        img_name = osp.basename(img_dir)
        with open(json_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img = cv2.imread(img_dir)
        joint_cor = []
        for shape in data['shapes']:
            cls = shape['label']
            points = shape['points']
            if cls == "joint":
                joint_cor.append(points)
            # if cls == "embedding":
            #     continue
            x1, y1 = points[0]
            x2, y2 = points[2]
            annotator = Annotator(img, line_width=2)
            annotator.box_label((x1, y1, x2, y2), label=f"{cls}", color=colors(label_map[cls], True))
        new_img = img[int(0):int(1344), int(joint_cor[0][0][0]):int(joint_cor[0][2][0])]
        cv2.imwrite(osp.join(dst_dir, img_name), new_img)

root_path = '/data/bt/xray_gangsi/LabeledData/zhoushan/target_joint/'
# vis_json('/data/bt/xray_gangsi/LabeledData/zhoushan/20240313_concat/images')
# folders = os.listdir(root_path)
folders = ['BC11A']
for folder in folders:
    folder_path = os.path.join(root_path, folder)
    vis_json(folder_path)
