# WRCD 数据集处理文档

## 1. 原始数据集结构

```
data/Wuhan road change detection dataset.nosync/
├── 2012image.tif               # 时相1：2012年遥感影像
├── 2014image.tif               # 时相2：2014年遥感影像
├── 2016image.tif               # 时相3：2016年遥感影像
├── 2012label.tif               # 2012年道路标签
├── 2014label.tif               # 2014年道路标签
├── 2016label.tif               # 2016年道路标签
├── change_label_2012to2014.tif # 2012→2014变化标签
└── change_label_2014to2016.tif # 2014→2016变化标签
```

原始图像尺寸：**10884 × 13655 像素**
覆盖区域：中国武汉市江夏区，面积约 194 km²，地面分辨率 0.2 m

---

## 2. 使用的库

| 库 | 用途 |
|----|------|
| **cv2** (OpenCV) | 图像读取、颜色空间转换、翻转增强 |
| **torch.utils.data** | PyTorch 数据集封装 |

核心代码位于 [Dataset.py](Dataset.py) 的 `read_directory()` 函数（第8-25行）

---

## 3. 预处理方式（来自论文 Section IV-A）

### 裁剪参数
| 参数 | 值 |
|------|-----|
| 裁剪尺寸 | **256 × 256** 像素 |
| 是否重叠 | **无重叠（nonoverlapping）** |
| 划分比例 | 训练集:测试集 = **7:3** |

### 样本数量
| 集合 | 样本数 |
|------|--------|
| 训练集 | 1560 对 |
| 测试集 | 666 对 |

> 论文原文："original images are cropped into **256 × 256 nonoverlapping patches**, and split into training and testing sets in a 7:3 ratio, resulting in **1560 training samples and 666 test samples**"

---

## 4. 裁剪计算验证

```
原始图像: 10884 × 13655
裁剪尺寸: 256 × 256

水平方向裁剪数量: 10884 / 256 = 42.5 → 42 块
垂直方向裁剪数量: 13655 / 256 = 53.3 → 53 块

总块数: 42 × 53 = 2226 块
训练集 (7:3): 2226 × 0.7 ≈ 1558 ≈ 1560
测试集 (3:10): 2226 × 0.3 ≈ 668 ≈ 666
```

> 注：实际处理时可能对边缘进行了适当裁剪或过滤

---

## 5. 处理后数据结构

```
CD_dataset/WRCD/
├── train/
│   ├── A/                    # 训练集时相1图像 (1560张)
│   ├── B/                    # 训练集时相2图像 (1560张)
│   └── label/                # 训练集变化标签 (1560张)
└── test/
    ├── A/                    # 测试集时相1图像 (666张)
    ├── B/                    # 测试集时相2图像 (666张)
    └── label/                # 测试集变化标签 (666张)
```

**注意**：Dataset.py 第140-148行显示 WRCD 使用与 CRCD 相同的路径配置：
```python
dataset_WRCD = '/opt/data/private/zq/Datasets/WRCD'
dataset_train_1 = '/train/A'
dataset_train_2 = '/train/B'
dataset_train_label = '/train/label'
dataset_test_1 = '/test/A'
dataset_test_2 = '/test/B'
dataset_test_label = '/test/label/'
```

---

## 6. 数据加载流程

### 6.1 图像读取 (`read_directory()`)
```python
def read_directory(directory_name, label=False):
    files = os.listdir(directory_name)
    files.sort(key=lambda x: int(x[0:-4]))  # 按文件名数字排序
    for filename in files:
        img = cv2.imread(directory_name + "/" + filename)
        if label:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 标签转灰度
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        array_of_img.append(img)
    return array_of_img
```

### 6.2 数据增强
```python
# 随机翻转增强
flipCote = random.choice([-1, 0, 1, 2])  # -1:水平+垂直, 0:垂直, 1:水平, 2:不翻转
imgs_1 = cv2.flip(imgs_1, flipCote)
imgs_2 = cv2.flip(imgs_2, flipCote)
label = cv2.flip(label, flipCote)
```

### 6.3 Dataset 类
```python
class LevirWhuGzDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return seq_img_1[index], seq_img_2[index], seq_label[index]
```

---

## 7. 生成 WRCD 数据集处理脚本

如果您需要自行处理原始 WRCD 数据集，可参考以下脚本：

```python
import os
import cv2
import numpy as np
from tqdm import tqdm

# 配置
RAW_DATA_DIR = "data/Wuhan road change detection dataset.nosync"
OUTPUT_DIR = "CD_dataset/WRCD"
PATCH_SIZE = 256
STRIDE = 256  # 无重叠，步长=块大小

def crop_and_save(image_path, output_dir, prefix, patch_size=256, stride=256):
    """裁剪大图像为小补丁"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return []

    h, w = img.shape[:2]
    filenames = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            idx = len(filenames)
            filename = f"{idx:04d}.tif"
            cv2.imwrite(os.path.join(output_dir, filename), patch)
            filenames.append(filename)

    return filenames

# 裁剪训练集/测试集（按7:3划分）
# ...
```

---

## 8. 注意事项

1. **裁剪尺寸**：代码仓库使用 **256×256**，而非 512×512
2. **无重叠**：STRIDE = PATCH_SIZE = 256
3. **路径配置**：实际 WRCD 路径需根据您的本地环境修改 `dataset_WRCD` 变量
4. **变化检测对**：使用时需提供两时相图像（如 2012image.tif 和 2014image.tif）及对应变化标签
