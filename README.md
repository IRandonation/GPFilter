# GPMPFilter - 高斯过程运动轨迹滤波器

## 项目简介

GPMPFilter是一个基于高斯过程（Gaussian Process）和因子图优化的运动轨迹滤波器。该项目使用GTSAM（Georgia Tech Smoothing and Mapping）库实现了一个完整的轨迹处理系统，能够对带有噪声的轨迹数据进行滤波和平滑处理，同时考虑运动学约束（如速度和加速度限制）。系统支持GP插值和时间参数化，实现了从带噪音的离散点到具体已经时间参数化的轨迹规划的完整链路。

## 核心功能

- **轨迹平滑**：使用高斯过程模型对原始轨迹数据进行滤波和平滑
- **GP插值**：将滤波后的数据点插值为10倍，以获得更平滑的轨迹
- **时间参数化**：基于速度和加速度确定整个轨迹的时间点以及总耗时
- **运动学约束**：支持速度和加速度约束，确保生成的轨迹符合物理规律
- **噪声抑制**：通过测量因子抑制观测噪声
- **可视化分析**：提供Python工具来比较原始轨迹、滤波轨迹、插值轨迹和时间参数化轨迹

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
│   ├── AccelerationConstraint.h # 加速度约束因子定义
│   ├── GPInterpolator.h   # GP插值器定义
│   
├── src/                   # C++源代码目录
│   ├── main.cpp           # 主程序实现
│   ├── GPInterpolator.cpp # GP插值器实现
│   
├── data/                  # 数据文件目录
│   ├── trajectory.csv     # 原始轨迹数据
│   └── trajectory_2d.csv  # 2D原始轨迹数据
├── output/                # 输出文件目录
│   ├── smoothed_trajectory.csv # 滤波后的轨迹数据
│   
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

### 5. GPInterpolator（GP插值器）

[`GPInterpolator.h`](include/GPInterpolator.h:1)和[`GPInterpolator.cpp`](src/GPInterpolator.cpp:1)实现了高斯过程插值功能，用于将滤波后的数据点插值为更高密度的轨迹点。

- **功能**：将滤波后的数据点插值为10倍，以获得更平滑的轨迹
- **插值方法**：使用高斯过程回归进行插值
- **核心算法**：
  - 状态转移矩阵计算
  - 过程噪声协方差计算
  - 高斯过程回归插值
- **参数**：
  - 原始时间步长（dt）
  - 插值因子（默认为10）

### 6. TimeParameterization（时间参数化器）

[`TimeParameterization.h`](include/TimeParameterization.h:1)和[`TimeParameterization.cpp`](src/TimeParameterization.cpp:1)实现了基于速度和加速度的时间参数化功能，用于确定整个轨迹的时间点以及总耗时。

- **功能**：基于速度和加速度确定整个轨迹的时间点以及总耗时
- **核心算法**：
  - 最小时间计算（基于速度和加速度约束）
  - 速度剖面生成（梯形或三角形速度剖面）
  - 三次样条插值
- **参数**：
  - 最大速度限制（maxVelocity）
  - 最大加速度限制（maxAcceleration）
  - 默认时间步长（defaultDt）
- **输出**：带时间戳的轨迹状态，包含位置和速度信息

## 速度和加速度的计算方法

### 速度计算

在GPMPFilter系统中，速度是通过以下方式计算和得到的：

1. **状态向量表示**：
   - 每个状态点使用4维向量表示：`[x, y, vx, vy]`
   - 其中`vx`和`vy`分别是x和y方向的速度分量

2. **速度大小计算**：
   - 速度大小通过欧几里得范数计算：`velocity = sqrt(vx² + vy²)`
   - 在[`main.cpp`](src/main.cpp:115)中的实现：
     ```cpp
     double velocity = std::sqrt(vx * vx + vy * vy);
     ```

3. **速度约束**：
   - 通过[`VelocityConstraint`](include/VelocityConstraint.h:1)类限制最大速度
   - 当速度超过阈值时，在优化过程中施加惩罚
   - 默认最大速度限制为5.0 m/s

4. **速度优化**：
   - 速度作为状态向量的一部分，通过因子图优化得到
   - GP过程因子确保相邻状态点之间的速度变化平滑
   - 测量因子间接影响速度估计，确保轨迹符合观测数据

### 加速度计算

加速度是通过以下方式计算和得到的：

1. **数值微分方法**：
   - 使用中心差分法计算加速度：`a = (v₂ - v₁) / (2 * dt)`
   - 在[`main.cpp`](src/main.cpp:117-126)中的实现：
     ```cpp
     // 计算加速度（使用中心差分法）
     double acceleration = 0.0;
     if (i > 0 && i < trajectory.size() - 1) {
         const auto& prevState = trajectory[i-1];
         const auto& nextState = trajectory[i+1];
         
         double ax = (nextState[2] - prevState[2]) / (2 * dt);
         double ay = (nextState[3] - prevState[3]) / (2 * dt);
         acceleration = std::sqrt(ax * ax + ay * ay);
     }
     ```

2. **加速度约束**：
   - 通过[`AccelerationConstraint`](include/AccelerationConstraint.h:1)类限制最大加速度
   - 加速度约束作用于相邻状态点之间：`a = (v₂ - v₁) / dt`
   - 当加速度超过阈值时，在优化过程中施加惩罚
   - 默认最大加速度限制为2.0 m/s²

3. **加速度在GP插值中的处理**：
   - 在[`GPInterpolator.cpp`](src/GPInterpolator.cpp:129-169)中，插值过程中会考虑加速度限制
   - 如果估计的加速度超过限制，使用更保守的插值方法：
     ```cpp
     // 估计加速度：基于速度变化和时间
     gtsam::Vector2 accel_estimate = (velocity2 - velocity1) / dt;
     double accel_magnitude = accel_estimate.norm();
     
     // 如果估计的加速度超过限制，则使用更保守的插值方法
     if (accel_magnitude > max_acceleration) {
         // 使用更平滑的插值方法：限制速度变化率
         double max_velocity_change = max_acceleration * interpolated_dt;
         // ... 限制速度变化的代码
     }
     ```

4. **自适应时间步长**：
   - 系统使用自适应时间步长来确保速度和加速度约束得到满足
   - 在[`main.cpp`](src/main.cpp:162-193)中计算自适应时间步长：
     ```cpp
     // 基于加速度约束计算最小时间
     double timeFromAcceleration = deltaVelMagnitude / maxAcceleration;
     
     // 基于速度约束计算最小时间
     double avgSpeed = (vel1.norm() + vel2.norm()) / 2.0;
     double timeFromDistance = distance / std::max(avgSpeed, 0.1);
     
     // 取两者中的较大值，确保同时满足速度和加速度约束
     double minTime = std::max(timeFromAcceleration, timeFromDistance);
     double adaptiveDt = std::max(minTime, minDt);
     ```

### 计算流程总结

1. **初始化阶段**：
   - 从CSV文件加载位置数据，初始速度设为0
   - 设置速度和加速度约束参数

2. **优化阶段**：
   - 使用因子图优化同时估计位置和速度
   - GP过程因子确保速度变化的平滑性
   - 速度和加速度约束因子确保运动学可行性

3. **插值阶段**：
   - 使用高斯过程回归进行插值
   - 在插值过程中考虑加速度限制，确保插值轨迹的平滑性

4. **输出阶段**：
   - 计算并保存每个点的速度和加速度信息
   - 输出统计信息，包括最大/平均速度和加速度

这种计算方法确保了生成的轨迹不仅在位置上平滑，而且在速度和加速度上也符合物理规律，适用于实际机器人运动控制等应用场景。

## 主要算法流程

[`main.cpp`](src/main.cpp:1)中的主程序实现了完整的轨迹处理流程，从带噪音的离散点到时间参数化的轨迹规划：

1. **数据加载**：从CSV文件加载原始轨迹数据
2. **因子图初始化**：创建非线性因子图和初始值估计
3. **因子添加**：
   - 先验因子：固定起点和终点
   - GP过程因子：建模动力学连续性
   - 测量因子：抑制观测噪声
   - 速度约束：限制最大速度
   - 加速度约束：限制最大加速度
4. **优化求解**：使用Levenberg-Marquardt优化器求解，获得滤波后的轨迹
5. **GP插值**：
   - 初始化GP插值器（插值因子为10）
   - 对滤波后的轨迹进行GP插值，获得更平滑的轨迹
6. **时间参数化**：
   - 初始化时间参数化器
   - 对插值后的平滑轨迹进行时间参数化
   - 计算轨迹总耗时
7. **结果保存**：
   - 保存滤波后的轨迹到CSV文件
   - 保存插值后的平滑轨迹到CSV文件
   - 保存时间参数化后的轨迹到CSV文件（包含时间戳、位置和速度信息）

## 数据格式

### 输入数据格式

原始轨迹数据（[`data/trajectory.csv`](data/trajectory.csv)）：
- 支持两种格式：
  - `x,y` 格式：包含x和y坐标
  - `x` 格式：仅包含x坐标（y坐标设为0）

### 输出数据格式

1. 滤波后的轨迹数据（[`output/smoothed_trajectory.csv`](output/smoothed_trajectory.csv)）：
   - `x,y` 格式：包含平滑后的x和y坐标

2. 插值后的轨迹数据（[`output/interpolated_trajectory.csv`](output/interpolated_trajectory.csv)）：
   - `x,y` 格式：包含插值后的x和y坐标

3. 时间参数化后的轨迹数据（[`output/time_parameterized_trajectory.csv`](output/time_parameterized_trajectory.csv)）：
   - `timestamp,x,y,vx,vy` 格式：包含时间戳、位置和速度信息
   - timestamp：时间戳（秒）
   - x, y：位置坐标
   - vx, vy：速度分量

## 轨迹比较工具

[`compare_trajectories.py`](compare_trajectories.py:1)提供了一个强大的轨迹比较工具，用于分析滤波效果：

### 功能特性

- **数据加载**：支持加载原始轨迹、滤波后轨迹、插值轨迹和时间参数化轨迹
- **统计分析**：计算误差统计信息（均值、标准差、最大/最小误差）
- **可视化**：
  - 轨迹对比图（支持多条轨迹同时显示）
  - 坐标对比图
  - 误差分析图
  - 速度剖面图（时间参数化数据）
  - 时间颜色编码轨迹图（时间参数化数据）
- **多数据支持**：可同时比较原始、滤波、插值和时间参数化轨迹

### 使用方法

```bash
python compare_trajectories.py [选项]
```

#### 选项参数

- `--original, -o`：原始轨迹数据文件路径（默认：data/trajectory.csv）
- `--smoothed, -s`：滤波后轨迹数据文件路径（默认：output/smoothed_trajectory.csv）
- `--original-2d, -2d`：2D原始轨迹数据文件路径（可选，默认：data/trajectory_2d.csv）
- `--interpolated, -i`：插值轨迹数据文件路径（可选）
- `--time-parameterized, -t`：时间参数化轨迹数据文件路径（可选，默认：output/time_parameterized_trajectory.csv）
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
   - 轨迹对比图（支持多条轨迹同时显示）
   - X坐标对比图
   - Y坐标对比图
   - 误差分析图
   - 速度剖面图（时间参数化数据）
   - 时间颜色编码轨迹图（时间参数化数据）

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