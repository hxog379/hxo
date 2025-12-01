# uav_crack_segmentation.py
import cv2
import torch
import numpy as np
from mmseg.apis import init_model, inference_model


# 初始化模型
def setup_model():
    config = 'configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py'
    checkpoint = 'checkpoints/deeplabv3_r50-d8_512x1024_40k_cityscapes.pth'
    model = init_model(config, checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu')
    return model


# 无人机图像预处理
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# 裂缝分割推理
def segment_cracks(model, image):
    result = inference_model(model, image)
    mask = result.pred_sem_seg.data[0].cpu().numpy()
    crack_mask = (mask == 1).astype(np.uint8) * 255
    return crack_mask


# 主函数
if __name__ == "__main__":
    # 初始化模型
    model = setup_model()

    # 输入和输出路径
    input_image = "test_image.jpg"
    output_mask = "crack_prediction.png"

    # 处理图像
    image = preprocess_image(input_image)
    crack_mask = segment_cracks(model, image)

    # 保存结果
    cv2.imwrite(output_mask, crack_mask)
    print(f"裂缝分割完成！结果保存在: {output_mask}")