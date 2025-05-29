# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import h5py
from PIL import Image
import glob
import traceback

# --------------- 定义模型 ---------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --------------- 配置参数 ---------------
input_size = 1024
hidden_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ********** 核心参数 **********
scaling_factor = 64    # <--- 每个原始坐标缩小 64 倍
block_size = 16        # <--- 每个缩小后的坐标代表一个 16x16 的块
# ****************************

model_name = "UNI"
target_labels = [0, 1,2,3,4]
num_classes = len(target_labels)


# ### 四分类
color_palette = {
    4: (0, 0, 0),        # 背景 - 黑色
    0: (250, 0,250),   # 类别 0  - 紫色
    1: (250, 0, 0),      # 类别 1 - 红色
    2: (0, 250, 0),      # 类别 2  - 绿色
    3: (0, 0, 250)       # 类别 3  - 蓝色

}

# ### 三分类
# color_palette = {
#     0: (0, 0, 0),        # 背景 - 黑色
#     1: (250, 0, 0),      # 类别 1 (原预测 0) - 红色
#     2: (30, 144, 255),      # 类别 2 (原预测 1) - 绿色
#     3: (255, 215, 0),      # 类别 3 (原预测 2) - 蓝色
# }

# ### 二分类
# color_palette = {
#     0: (0, 0, 0),        # 背景 - 黑色
#     1: (51, 51, 254),      # 类别 1 (原预测 0) - 红色
#     2: (255, 51, 255),      # 类别 2 (原预测 1) - 绿色
# }


default_color = (255, 255, 255)

model_paths = [f"/home/gjs/ESD_2025/Experiment/ckpt/{model_name}/{num_classes}cls_best_fold_{i+1}.pth" for i in range(5)]
data_root_dir = f"/data_nas2/gjs/ESD_2025/classification/{model_name}_{num_classes}cls/5_fold/"
output_dir_base = f"./{model_name}_{num_classes}cls_predmaps_scaled{scaling_factor}_block{block_size}_v2" # 新目录名
os.makedirs(output_dir_base, exist_ok=True)

print(f"使用设备: {device}")
print(f"模型预测类别数: {num_classes}")
print(f"目标标签: {target_labels}")
print(f"坐标缩放因子: {scaling_factor}")
print(f"最终图块大小: {block_size}x{block_size}")
print(f"颜色映射: {color_palette}")

# --------------- 评估和预测图生成 ---------------
for fold in range(5):
    fold_num = fold + 1
    print(f"\n--- 正在处理 Fold {fold_num} ---")

    # --- 加载模型 ---
    model_path = model_paths[fold]
    if not os.path.exists(model_path):
        print(f"警告: 未找到 Fold {fold_num} 模型路径: {model_path}")
        continue
    model = MLP(input_size, hidden_size, num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"错误: 加载 Fold {fold_num} 模型状态时出错: {e}")
        continue
    model.eval()
    print(f"已加载模型: {model_path}")

    # --- 查找 H5 文件 ---
    h5_dir = os.path.join(data_root_dir, f"fold_{fold_num}", "val")
    if not os.path.isdir(h5_dir):
        print(f"警告: 未找到 Fold {fold_num} H5 目录: {h5_dir}")
        continue
    h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))
    if not h5_files:
        print(f"警告: 在 {h5_dir} 未找到 H5 文件")
        continue
    print(f"找到 {len(h5_files)} 个 H5 文件")

    # --- 创建 Fold 输出目录 ---
    # fold_output_dir = os.path.join(output_dir_base, f"fold_{fold_num}")
    fold_output_dir = output_dir_base
    os.makedirs(fold_output_dir, exist_ok=True)

    # --- 处理每个 H5 文件 ---
    for h5_path in h5_files:
        h5_filename = os.path.basename(h5_path)
        print(f"  处理文件: {h5_filename}")

        try:
            # --- 数据加载与检查 ---
            with h5py.File(h5_path, 'r') as f:
                if not all(k in f for k in ['coords', 'features', 'labels']):
                    print(f"    警告: 跳过 {h5_filename} - 缺少数据集。")
                    continue
                coords = f['coords'][()]
                features = f['features'][()]
                labels = f['labels'][()]
                n_patches = coords.shape[0]
                if n_patches == 0:
                    print(f"    警告: 跳过 {h5_filename} - 无数据。")
                    continue
                if not (features.shape[0] == n_patches and labels.shape[0] == n_patches):
                     print(f"    警告: 跳过 {h5_filename} - 数据集维度不匹配。")
                     continue

            # --- 筛选需要预测的数据 ---
            valid_mask = np.isin(labels, target_labels)
            filtered_coords = coords[valid_mask]
            filtered_features = features[valid_mask]
            num_valid_patches = filtered_features.shape[0]

            if num_valid_patches == 0:
                print(f"    信息: {h5_filename} 中无有效标签 patch。")
                # 仍需计算画布大小以生成可能的空图或背景图

            # --- 模型预测 ---
            predictions = np.array([], dtype=int)
            if num_valid_patches > 0:
                with torch.no_grad():
                    features_tensor = torch.tensor(filtered_features, dtype=torch.float32).to(device)
                    outputs = model(features_tensor)
                    pred_indices = torch.argmax(outputs, dim=1)
                    predictions = pred_indices.cpu().numpy()

            # --- 计算最终图像尺寸 ---
            # 使用 *所有* 原始坐标来确定边界
            if n_patches > 0: # 确保coords不为空
                # 将所有原始坐标缩小 scaling_factor 倍
                scaled_all_coords = coords // scaling_factor
                # 找到缩小后坐标的最大值
                max_scaled_x = np.max(scaled_all_coords[:, 0])
                max_scaled_y = np.max(scaled_all_coords[:, 1])
                # 最终图像的宽度和高度需要覆盖到最右下角块的边缘
                final_map_width = max_scaled_x + block_size
                final_map_height = max_scaled_y + block_size
            else: # 如果文件为空，创建最小画布
                final_map_width = block_size
                final_map_height = block_size


            # --- 创建最终的彩色图 ---
            # 创建空白最终彩色图 (RGB, uint8), 背景默认为黑色 (0, 0, 0)
            final_color_map = np.zeros((final_map_height, final_map_width, 3), dtype=np.uint8)

            # --- 填充最终彩色图 ---
            if num_valid_patches > 0:
                '''  3分类/2分类
                # 将预测标签值 +1 (从 1 开始)
                predicted_labels_plus_one = predictions + 1
                '''
                #5分类
                predicted_labels_plus_one = predictions
                
                # 遍历每一个 *有效* 的 patch
                for i in range(num_valid_patches):
                    # 获取当前 patch 的原始坐标
                    original_coord = filtered_coords[i]
                    # 计算其在最终图上的块的左上角坐标
                    x_start = original_coord[0] // scaling_factor
                    y_start = original_coord[1] // scaling_factor
                    # 计算块的结束坐标 (不包含)
                    x_end = x_start + block_size
                    y_end = y_start + block_size

                    # 获取预测标签对应的颜色
                    label_value = predicted_labels_plus_one[i]
                    color = color_palette.get(label_value, default_color) # 使用 get 获取颜色，提供默认值

                    # 确保坐标不越界 (虽然理论上 final_map 应该足够大)
                    y_end = min(y_end, final_map_height)
                    x_end = min(x_end, final_map_width)

                    # 使用切片填充颜色块
                    final_color_map[y_start:y_end, x_start:x_end] = color

            # --- 保存最终的彩色预测图 ---
            # output_filename = h5_filename.replace('.h5', f'_predmap_s{scaling_factor}_b{block_size}_v2.png')
            output_filename = h5_filename.replace('.h5', f'.png')
            output_png_path = os.path.join(fold_output_dir, output_filename)
            try:
                pil_img = Image.fromarray(final_color_map)
                pil_img.save(output_png_path)
                print(f"    已将最终彩色预测图 ({num_valid_patches}/{n_patches} 有效 patch) 保存至: {output_png_path}")
            except Exception as e:
                print(f"    错误: 保存最终彩色图像 {output_png_path} 时出错: {e}")

        except Exception as e:
            print(f"  错误: 处理文件 {h5_filename} 时出错: {e}")
            traceback.print_exc()

print("\n--- 已完成所有 v2 彩色预测图生成 ---")