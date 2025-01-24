'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2025-01-20 06:49:03
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2025-01-21 01:51:12
Description: 
'''
import numpy as np
import cv2

def read_masks_from_file(mask_file):
    with open(mask_file, 'rb') as f:
        # 读取 mask 数量
        num_masks = np.fromfile(f, dtype=np.uint64, count=1)[0]
        masks = []
        
        for _ in range(num_masks):
            # 读取每个 mask 的维度
            rows = np.fromfile(f, dtype=np.int32, count=1)[0]
            cols = np.fromfile(f, dtype=np.int32, count=1)[0]
            
            # 读取每个 mask 的数据
            mask_data = np.fromfile(f, dtype=np.float32, count=rows * cols)
            mask = mask_data.reshape((rows, cols))
            masks.append(mask)
    
    return masks

def visualize_masks_on_single_image(masks, original_image_path):
    # 加载原图像
    original_image = cv2.imread(original_image_path)
    # 确保原图像加载成功
    if original_image is None:
        print(f"Error: Unable to load image at {original_image_path}")
        return

    # 初始化一个全黑的图像，用来叠加所有的 mask
    overlay_image = np.zeros_like(original_image)
    

    # 遍历每个 mask 并叠加到 overlay_image 上
    for i, mask in enumerate(masks):
        # 将 mask 转换为二值图
        _, binary_mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 创建一个与原图相同尺寸的全黑图像
        colored_mask = np.zeros_like(original_image)
        colored_mask[:, :, 2] = binary_mask * 255  # 仅在红色通道显示

        # 将该 mask 叠加到叠加图像上
        overlay_image = cv2.addWeighted(overlay_image, 1, colored_mask, 0.5, 0)
        overlay_image_copy = np.zeros_like(original_image)
        overlay_image_copy = cv2.addWeighted(overlay_image_copy, 1, colored_mask, 0.5, 0)
        cv2.imwrite(f"{i}.jpg", overlay_image_copy)

    # 将叠加的 mask 和原图进行合成
    final_image = cv2.addWeighted(original_image, 0.7, overlay_image, 0.3, 0)

    # 显示最终合成图像
    cv2.imwrite("final_image.jpg", final_image)
    # cv2.imshow("All Masks Overlay", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 使用示例
mask_file = '/data/Datasets/public/coco_instanceseg/debug-test/output/000000442993.bin'
original_image_path = '/data/Datasets/public/coco_instanceseg/debug-test/000000442993.jpg'

masks = read_masks_from_file(mask_file)
visualize_masks_on_single_image(masks, original_image_path)
