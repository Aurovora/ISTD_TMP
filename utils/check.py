# 这是一个简单的可视化检查脚本，放在你的 dataloader_for_DAUB.py 同级目录下运行
import torch
import numpy as np
import cv2
from dataloader_for_DAUB import seqDataset, dataset_collate
from torch.utils.data import DataLoader

# 1. 配置路径
# DATA_ROOT = "D:/affair/college/ISTD/code/TMP-main/dataDAUB"
ANNOTATION_PATH = "D:/affair/college/ISTD/TMP/DAUB/coco_train_DAUB.txt" # 确保这是你生成的新txt


# 2. 实例化 DataLoader
dataset = seqDataset(ANNOTATION_PATH, 512, 5, 'train')
gen = DataLoader(dataset, shuffle=True, batch_size=4, collate_fn=dataset_collate)

# 3. 取出一个 Batch 进行可视化
print("正在获取一个 Batch 的数据...")
images, bboxes = next(iter(gen))

# images shape: [Batch, Channel, Frames, H, W] -> [4, 3, 5, 512, 512]
# bboxes is a list of tensors

# 4. 反归一化 (Un-normalize) 以便显示
# 你的 preprocess 是: image /= 255.0; image -= mean; image /= std
def de_process(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy() # C,H,W -> H,W,C
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img.copy() # 必须copy才能画图

# 5. 画图检查
for b in range(len(images)): # 遍历 Batch 中的每组样本
    # 只看最后一帧 (关键帧)
    # images[b] shape: [3, 5, 512, 512] -> 取第4个索引即第5帧
    current_frame = images[b][:, 4, :, :] 
    img_vis = de_process(current_frame)
    
    # 获取该样本的框
    current_boxes = bboxes[b] # shape: [N, 5] (x, y, x, y, c)
    
    print(f"Sample {b}: 发现 {len(current_boxes)} 个目标")
    for box in current_boxes:
        xc, yc, w, h, c = box.tolist() 
        
        # ⚠️ 2. 转换为角点坐标 (xmin, ymin, xmax, ymax)
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)
        
        # ⚠️ 3. 打印转换后的角点坐标 (方便检查 Letterbox)
        print(f"  - 角点 Box: {x1}, {y1}, {x2}, {y2}. 类别: {int(c)}")
        
        # 4. 画框 (绿色)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img_vis, str(int(c)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # 可选：打印类别

    # 保存或显示
    save_path = f"debug_check_{b}.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    print(f"  已保存可视化图片到: {save_path}")

print("\n请打开生成的 debug_check_x.jpg 图片：")
print("1. 图片是否清晰？(颜色是否怪异？)")
print("2. 绿色的框是否准确框住了目标？")
print("3. 如果没有绿框，或者框歪了，说明 DataLoader 或 txt 有问题！")