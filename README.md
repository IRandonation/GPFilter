# GPMPFilter - 高斯过程运动轨迹滤波器

## 项目简介

GPMPFilter是一个基于高斯过程（Gaussian Process）和因子图优化的运动轨迹滤波器。该项目使用GTSAM（Georgia Tech Smoothing and Mapping）库实现了一个完整的轨迹平滑系统，能够对带有噪声的轨迹数据进行滤波和平滑处理，同时考虑运动学约束（如速度和加速度限制）。

## 核心功能

- **轨迹平滑**：使用高斯过程模型对原始轨迹数据进行滤波和平滑
- **运动学约束**：支持速度和加速度约束，确保生成的轨迹符合物理规律
- **噪声抑制**：通过测量因子抑制观测噪声
- **可视化分析**：提供Python工具来比较原始轨迹和滤波后的轨迹

## 项目结构

```
GPMPFilter/
├── CMakeLists.txt          # CMake构建配置文件
├── README.md              # 项目说明文档
├── pyproject.toml         # Python项目配置文件
├── main.py                # Python主程序（简单示例）
├── compare_trajectories.py # 轨迹比较工具
├── include/               # C++头文件目录
│   ├── GPFactor.h         # 高斯过程因子定义
│   ├── MeasurementFactor.h # 测量因子定义
│   ├── VelocityConstraint.h # 速度约束因子定义
│   └── AccelerationConstraint.h # 加速度约束因子定义
├── src/                   # C++源代码目录
│   └── main.cpp           # 主程序实现
├── data/                  # 数据文件目录
│   ├── trajectory.csv     # 原始轨迹数据
│   └── trajectory_2d.csv  # 2D原始轨迹数据
├── output/                # 输出文件目录
│   └── smoothed_trajectory.csv # 滤波后的轨迹数据
└── build/                 # 构建目录
```

## 核心组件

### 1. GPFactor（高斯过程因子）

[`GPFactor.h`](include/GPFactor.h:1)定义了高斯过程因子，用于建模轨迹的动力学连续性。该因子连接相邻时间步的状态变量，确保轨迹的平滑性。

- **功能**：建模状态转移动力学，确保轨迹的连续性
- **数学模型**：使用状态转移矩阵F_连接相邻状态
- **误差计算**：`x2 - F_ * x1`

### 2. MeasurementFactor（测量因子）

[`MeasurementFactor.h`](include/MeasurementFactor.h:1)定义了测量因子，用于将观测数据整合到因子图中。

- **功能**：将观测位置数据整合到优化中
- **误差计算**：观测位置与状态位置的差异
- **输出**：2维位置误差（x, y坐标）

### 3. VelocityConstraint（速度约束）

[`VelocityConstraint.h`](include/VelocityConstraint.h:1)定义了速度约束因子，用于限制轨迹的最大速度。

- **功能**：确保轨迹速度不超过物理限制
- **约束机制**：当速度超过阈值时施加惩罚
- **参数**：最大允许速度（m/s）

### 4. AccelerationConstraint（加速度约束）

[`AccelerationConstraint.h`](include/AccelerationConstraint.h:1)定义了加速度约束因子，用于限制轨迹的最大加速度。

- **功能**：确保轨迹加速度不超过物理限制
- **约束机制**：当加速度超过阈值时施加惩罚
- **参数**：最大允许加速度（m/s²）

## 主要算法流程

[`main.cpp`](src/main.cpp:1)中的主程序实现了完整的轨迹滤波流程：

1. **数据加载**：从CSV文件加载原始轨迹数据
2. **因子图初始化**：创建非线性因子图和初始值估计
3. **因子添加**：
   - 先验因子：固定起点和终点
   - GP过程因子：建模动力学连续性
   - 测量因子：抑制观测噪声
   - 速度约束：限制最大速度
   - 加速度约束：限制最大加速度
4. **优化求解**：使用Levenberg-Marquardt优化器求解
5. **结果保存**：将滤波后的轨迹保存到CSV文件

## 数据格式

### 输入数据格式

原始轨迹数据（[`data/trajectory.csv`](data/trajectory.csv)）：
- 支持两种格式：
  - `x,y` 格式：包含x和y坐标
  - `x` 格式：仅包含x坐标（y坐标设为0）

### 输出数据格式

滤波后的轨迹数据（[`output/smoothed_trajectory.csv`](output/smoothed_trajectory.csv)）：
- `x,y` 格式：包含平滑后的x和y坐标

## 轨迹比较工具

[`compare_trajectories.py`](compare_trajectories.py:1)提供了一个强大的轨迹比较工具，用于分析滤波效果：

### 功能特性

- **数据加载**：支持加载原始轨迹和滤波后轨迹
- **统计分析**：计算误差统计信息（均值、标准差、最大/最小误差）
- **可视化**：生成轨迹对比图、坐标对比图和误差分析图
- **多数据支持**：可同时比较2D和4D轨迹数据

### 使用方法

```bash
python compare_trajectories.py [选项]
```

#### 选项参数

- `--original, -o`：原始轨迹数据文件路径（默认：data/trajectory.csv）
- `--smoothed, -s`：滤波后轨迹数据文件路径（默认：output/smoothed_trajectory.csv）
- `--original-2d, -2d`：2D原始轨迹数据文件路径（可选，默认：data/trajectory_2d.csv）
- `--save-plot, -p`：保存图像到指定路径

#### 示例

```bash
# 使用默认设置
python compare_trajectories.py

# 指定自定义文件路径
python compare_trajectories.py --original my_data.csv --smoothed my_output.csv

# 保存图像到指定路径
python compare_trajectories.py --save-plot comparison_result.png
```

### 输出内容

1. **统计信息**：
   - X方向误差（均值、标准差）
   - Y方向误差（均值、标准差）
   - 误差幅值（均值、标准差、最大值、最小值）

2. **可视化图表**：
   - 轨迹对比图
   - X坐标对比图
   - Y坐标对比图
   - 误差分析图

## 构建和运行

### 依赖项

- **GTSAM**：Georgia Tech Smoothing and Mapping库
- **Eigen3**：线性代数库
- **CMake**：构建系统
- **Python**：用于轨迹比较工具

### 构建步骤

1. 创建构建目录：
```bash
mkdir build
cd build
```

2. 运行CMake：
```bash
cmake ..
```

3. 编译项目：
```bash
make
```

### 运行程序

1. 设置库路径：
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

2. 运行滤波器：
```bash
./GPMPFilter
```

3. 运行轨迹比较工具：
```bash
python compare_trajectories.py
```

## 参数配置

### 滤波器参数

在[`main.cpp`](src/main.cpp:98)中可以调整以下参数：

- `dt`：时间步长（默认：0.1秒）
- `prior_noise`：先验噪声模型
- `gp_noise`：高斯过程噪声模型
- `meas_noise`：测量噪声模型
- `vel_noise`：速度约束噪声模型
- `accel_noise`：加速度约束噪声模型

### 约束参数

- **最大速度**：在[`VelocityConstraint`](include/VelocityConstraint.h:14)中设置（默认：5.0 m/s）
- **最大加速度**：在[`AccelerationConstraint`](include/AccelerationConstraint.h:14)中设置（默认：2.0 m/s²）

## 扩展和定制

### 添加新的约束因子

1. 在`include/`目录下创建新的头文件
2. 继承自`gtsam::NoiseModelFactor1<gtsam::Vector4>`或适当的基类
3. 实现`evaluateError`方法
4. 在[`main.cpp`](src/main.cpp:1)中添加新的因子到因子图

### 修改噪声模型

可以通过调整噪声模型的参数来改变不同因子的权重：

```cpp
// 示例：调整先验噪声
auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector(4) << 0.06, 0.06, 0.5, 0.5).finished());
```

## 应用场景

- **机器人路径规划**：平滑机器人运动轨迹
- **自动驾驶**：车辆轨迹滤波和预测
- **运动分析**：人体运动数据处理
- **信号处理**：时序数据平滑和去噪

## 许可证

该项目目前未指定许可证。

## 贡献

欢迎提交问题和改进建议！

## 联系方式

如有问题或建议，请通过GitHub Issues联系。