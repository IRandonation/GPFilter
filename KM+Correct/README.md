# 轨迹数据卡尔曼滤波与渐进校正

本实现提供了一个完整的卡尔曼滤波器解决方案，用于轨迹数据处理，并结合渐进校正以平滑位置误差。

## 概述

该系统通过两个主要阶段处理轨迹数据：

1.  **卡尔曼滤波**：从有噪声的位置测量中估计位置、速度和加速度。
2.  **渐进校正**：在多个帧上平滑地施加误差校正，以避免突然跳变。

## 文件说明

-   `src/kalman_filter.py`：核心卡尔曼滤波器实现，包含渐进校正逻辑。
-   `src/visualize_results.py`：用于比较不同数据集的可视化工具。
-   `src/main.py`：运行整个流程的主执行脚本。
-   `README.md`：此文档文件。

## 生成的输出文件

-   `output/data1_filtered.csv`：包含位置、速度和加速度的滤波数据（18列）。
-   `output/data2_corrected.csv`：仅包含渐进校正后的位置数据（6列）。

## 生成的图表

-   `plots/position_comparison.png`：原始、滤波、校正和积分位置的比较。
-   `plots/velocity_comparison.png`：不同速度估计的比较。
-   `plots/acceleration_comparison.png`：不同加速度估计的比较。
-   `plots/error_analysis.png`：不同方法之间的误差分析。

## 使用方法

### 快速开始

运行完整的流程：

```bash
cd /d/Project/JD_Robot/RobotArm/GPFilter/KM+RTS
uv run python src/main.py
```

### 独立组件

#### 1. 卡尔曼滤波处理

```python
from src.kalman_filter import process_trajectory_data

data1_file, data2_file = process_trajectory_data(
    input_file="../data/trajectory.csv",
    output_dir="../output",
    dt=0.001  # 1ms 时间步长
)
```

#### 2. 可视化

```python
from src.visualize_results import TrajectoryVisualizer

visualizer = TrajectoryVisualizer(
    original_file="../data/trajectory.csv",
    data1_file="../output/data1_filtered.csv", 
    data2_file="../output/data2_corrected.csv"
)

visualizer.generate_all_plots("../plots")
```

## 算法详情

### 卡尔曼滤波器

-   **状态向量**：18维 [位置, 速度, 加速度]，用于6个自由度。
-   **运动模型**：恒定加速度模型。
-   **测量值**：仅位置（6个自由度）。
-   **时间步长**：1ms（可配置）。

### 渐进校正

-   **目的**：平滑滤波位置和速度积分位置之间的积分误差。
-   **方法**：将位置误差分布在N帧（默认：50帧）上。
-   **分布**：线性或指数衰减（可配置）。
-   **阈值**：校正激活的最小位置误差阈值（默认：0.001米）。

## 数据格式

### 输入数据 (trajectory.csv)
```
x, y, z, rx, ry, rz
```
-   位置坐标 (x, y, z)，单位为米。
-   旋转坐标 (rx, ry, rz)，单位为弧度。

### 输出数据1 (data1_filtered.csv)
```
x, y, z, rx, ry, rz, vx, vy, vz, vrx, vry, vrz, ax, ay, az, arx, ary, arz
```
-   所有6个自由度的滤波位置、速度和加速度。

### 输出数据2 (data2_corrected.csv)
```
x, y, z, rx, ry, rz
```
-   仅渐进校正后的位置数据。

## 可视化比较

### 位置比较
-   原始位置
-   数据1位置（滤波后）
-   数据2位置（校正后）
-   数据1速度积分位置

### 速度比较
-   原始差分速度
-   数据1速度（滤波后）
-   数据1位置差分速度
-   数据2差分速度

### 加速度比较
-   原始差分加速度
-   数据1加速度（滤波后）
-   数据1速度差分加速度
-   数据2差分加速度

## 配置参数

### 卡尔曼滤波器参数
-   `dt`：时间步长（默认：0.001秒）
-   `process_noise_pos`：位置过程噪声方差（默认：0.01）
-   `process_noise_vel`：速度过程噪声方差（默认：0.01）
-   `process_noise_acc`：加速度过程噪声方差（默认：0.01）
-   `measurement_noise`：测量噪声方差（默认：0.1）

### 渐进校正参数
-   `correction_frames`：误差分布的帧数（默认：50）
-   `correction_type`：'linear'（线性）或 'exponential'（指数）衰减
-   `error_threshold`：校正激活的最小误差（默认：0.001米）

## 依赖项

-   numpy
-   pandas
-   matplotlib
-   logging (内置)

## 注意事项

-   所有图表中的文本均使用英文，以避免字符渲染问题。
-   系统使用 uv 进行 Python 环境管理。
-   处理时间与输入数据大小呈线性关系。
-   内存使用针对实时处理进行了优化。