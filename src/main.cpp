// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Vector.h>

// 定义6维状态向量类型: [x, y, z, roll, pitch, yaw]
using Vector6 = gtsam::Vector;

#include "GPFactor.h"
#include "MeasurementFactor.h"
#include "GPInterpolator.h"

// 加载CSV数据文件
std::vector<Vector6> loadCSV(const std::string& filename) {
    std::vector<Vector6> measurements;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return measurements;
    }
    
    std::string line;
    
    // 读取所有位姿数据
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> values;
        double value;
        char comma;
        
        // 读取所有数值，用逗号分隔
        while (iss >> value) {
            values.push_back(value);
            if (iss.peek() == ',') {
                iss.ignore();
            }
        }
        
        // 创建6维状态向量 [x, y, z, roll, pitch, yaw]
        Vector6 state = Vector6::Zero(6);
        
        // 根据数据维度设置状态
        if (values.size() >= 6) {
            // 6自由度数据：x, y, z, roll, pitch, yaw
            state[0] = values[0]; // x
            state[1] = values[1]; // y
            state[2] = values[2]; // z
            state[3] = values[3]; // roll
            state[4] = values[4]; // pitch
            state[5] = values[5]; // yaw
        } else if (values.size() >= 3) {
            // 3自由度数据：x, y, z
            state[0] = values[0]; // x
            state[1] = values[1]; // y
            state[2] = values[2]; // z
            // roll, pitch, yaw 默认为0
        } else if (values.size() >= 2) {
            // 2自由度数据：x, y
            state[0] = values[0]; // x
            state[1] = values[1]; // y
            // z, roll, pitch, yaw 默认为0
        } else if (values.size() >= 1) {
            // 1自由度数据：x
            state[0] = values[0]; // x
            // y, z, roll, pitch, yaw 默认为0
        }
        
        measurements.push_back(state);
    }
    file.close();
    
    std::cout << "检测到 " << measurements.size() << " 个位姿点，维度: ";
    if (measurements.size() > 0) {
        bool hasRotation = (measurements[0][3] != 0.0 || measurements[0][4] != 0.0 || measurements[0][5] != 0.0);
        bool hasZ = (measurements[0][2] != 0.0);
        bool hasY = (measurements[0][1] != 0.0);
        
        if (!hasRotation) {
            if (!hasZ) {
                if (!hasY) {
                    std::cout << "1D (x)";
                } else {
                    std::cout << "2D (x, y)";
                }
            } else {
                std::cout << "3D (x, y, z)";
            }
        } else {
            std::cout << "6D (x, y, z, roll, pitch, yaw)";
        }
    }
    std::cout << std::endl;
    
    return measurements;
}

// 保存滤波后的轨迹
void saveTrajectory(const gtsam::Values& result, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }
    
    // 创建输出目录
    system("mkdir -p output");
    
    // 写入CSV头
    file << "x,y,z,roll,pitch,yaw" << std::endl;
    
    for (const auto& key_value : result) {
        gtsam::Key key = key_value.key;
        // 直接使用整数键值，因为我们使用的是整数索引
        Vector6 state = result.at<Vector6>(key);
        // 现在state包含位置和姿态信息
        file << state[0] << "," << state[1] << "," << state[2] << ","  // x, y, z
             << state[3] << "," << state[4] << "," << state[5] << std::endl;  // roll, pitch, yaw
    }
    
    file.close();
    std::cout << "滤波后的轨迹已保存到: " << filename << std::endl;
}


// 将Values转换为Vector6向量
std::vector<Vector6> valuesToVector(const gtsam::Values& result) {
    std::vector<Vector6> trajectory;
    
    for (const auto& key_value : result) {
        gtsam::Key key = key_value.key;
        Vector6 state = result.at<Vector6>(key);
        trajectory.push_back(state);
    }
    
    return trajectory;
}

// 保存Vector6向量到CSV文件（包含时间戳、位置和姿态）
void saveTrajectoryFromVector(const std::vector<Vector6>& trajectory, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }
    
    // 创建输出目录
    system("mkdir -p output");
    
    // 写入CSV头
    file << "x,y,z,roll,pitch,yaw" << std::endl;
    
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& state = trajectory[i];
        
        // 位置和姿态
        double x = state[0];
        double y = state[1];
        double z = state[2];
        double roll = state[3];
        double pitch = state[4];
        double yaw = state[5];
        
        file << x << "," << y << "," << z << ","
             << roll << "," << pitch << "," << yaw << std::endl;
    }
    
    file.close();
    std::cout << "插值后的轨迹已保存到: " << filename << std::endl;
}


int main() {
    // 1. 读取轨迹数据
    std::vector<Vector6> measurements = loadCSV("/home/chen/Documents/GPMPFilter/data/trajectory.csv");
    std::cout << "加载了 " << measurements.size() << " 个测量点" << std::endl;
    
    if (measurements.empty()) {
        std::cerr << "没有加载到数据，程序退出" << std::endl;
        return 1;
    }

    // 2. 初始化因子图
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    
    // 3. 添加初始值
    for (size_t i = 0; i < measurements.size(); ++i) {
        initial.insert(i, measurements[i]);
    }
    
    // 4. 添加因子
    // 使用固定时间步长
    const double dt = 0.1;  // 固定时间步长 (s)
    
    // 先验因子噪声：位置和姿态噪声
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.5, 0.5, 0.5).finished());  // 位置姿态噪声
    
    // 起点/终点约束（先验因子）
    graph.add(gtsam::PriorFactor<Vector6>(0, measurements[0], prior_noise));
    graph.add(gtsam::PriorFactor<Vector6>(measurements.size()-1, measurements.back(), prior_noise));
    
    // GP过程因子（建模动力学连续性）
    auto gp_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());  // 位置姿态约束中等
    for (size_t i = 0; i < measurements.size()-1; ++i) {
        // 使用固定时间步长
        graph.add(std::make_shared<GPFactor>(i, i+1, dt, gp_noise));
    }
    
    // 测量因子（抑制观测噪声）
    auto meas_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1);  // 6维测量噪声（位置和姿态）
    for (size_t i = 1; i < measurements.size()-1; ++i) {
        gtsam::Point3 position(measurements[i][0], measurements[i][1], measurements[i][2]);
        gtsam::Rot3 rotation = gtsam::Rot3::Yaw(measurements[i][5]) *
                              gtsam::Rot3::Pitch(measurements[i][4]) *
                              gtsam::Rot3::Roll(measurements[i][3]);
        gtsam::Pose3 pose(rotation, position);
        graph.add(std::make_shared<MeasurementFactor>(i, pose, meas_noise));
    }

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 输出初始误差信息
    std::cout << "=== 初始误差分析 ===" << std::endl;
    std::cout << "初始总误差: " << graph.error(initial) << std::endl;
    
    // 5. 优化求解
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosity("TERMINATION");
    params.setMaxIterations(1000);  // 增加最大迭代次数
    params.setRelativeErrorTol(1e-2);  // 设置相对误差容差
    params.setAbsoluteErrorTol(1e-1);  // 设置绝对误差容差
    params.setlambdaInitial(1e-5);  // 设置较小的初始lambda值
    params.setlambdaFactor(2.0);  // 设置lambda增长因子
    params.setlambdaUpperBound(1e10);  // 设置lambda上界
    params.setlambdaLowerBound(1e-10);  // 设置lambda下界
    params.setUseFixedLambdaFactor(false);  // 允许动态调整lambda
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    gtsam::Values result = optimizer.optimize();
    
    // 6. 保存滤波后的轨迹
    saveTrajectory(result, "/home/chen/Documents/GPMPFilter/output/smoothed_trajectory.csv");
    
    // 7. 将优化结果转换为Vector6向量
    std::vector<Vector6> smoothedTrajectory = valuesToVector(result);
    
    // 8. 初始化GP插值器
    const int interpolationFactor = 10; // 插值因子，将滤波后的数据点插值为10倍
    
    // 使用固定时间步长作为插值器的基础时间步长
    GPInterpolator interpolator(dt, interpolationFactor);

    interpolator.setLengthScale(dt * 3.0);  // 特征长度：设为时间步的1倍，平衡平滑性与跟踪性
    interpolator.setNoiseSigma(0.01);      // 观测噪声：减小噪声，使插值更贴近原始滤波轨迹
    
    // 9. 对滤波后的轨迹进行GP插值（仅用于获得平滑轨迹）
    std::vector<Vector6> interpolatedTrajectory = interpolator.interpolate(smoothedTrajectory);
    std::cout << "GP插值后得到 " << interpolatedTrajectory.size() << " 个数据点" << std::endl;
    
    // 10. 保存插值后的平滑轨迹（位置和姿态）
    saveTrajectoryFromVector(interpolatedTrajectory, "/home/chen/Documents/GPMPFilter/output/interpolated_trajectory.csv");
    
    // 输出固定时间步长信息
    std::cout << "=== 固定时间步长信息 ===" << std::endl;
    std::cout << "固定时间步长: " << dt << " s" << std::endl;
    
    // 16. 输出一些统计信息
    std::cout << "优化完成！" << std::endl;
    std::cout << "初始误差: " << graph.error(initial) << std::endl;
    std::cout << "最终误差: " << graph.error(result) << std::endl;
    std::cout << "原始数据点数: " << measurements.size() << std::endl;
    std::cout << "滤波后数据点数: " << smoothedTrajectory.size() << std::endl;
    std::cout << "插值后数据点数: " << interpolatedTrajectory.size() << std::endl;
    
    // 输出文件保存信息
    std::cout << std::endl;
    std::cout << "=== 输出文件 ===" << std::endl;
    std::cout << "滤波后的轨迹: output/smoothed_trajectory.csv" << std::endl;
    std::cout << "插值后的轨迹: output/interpolated_trajectory.csv" << std::endl;
    
    // 结束计时并输出总运行时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << std::endl;
    std::cout << "=== 计算时间统计 ===" << std::endl;
    std::cout << "程序总运行时间: " << duration.count() << " 毫秒" << std::endl;
    
    return 0;
}