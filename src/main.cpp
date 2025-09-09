// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/base/Vector.h>
#include "GPFactor.h"
#include "MeasurementFactor.h"
#include "VelocityConstraint.h"
#include "AccelerationConstraint.h"
#include "GPInterpolator.h"

// 加载CSV数据文件
std::vector<gtsam::Vector4> loadCSV(const std::string& filename) {
    std::vector<gtsam::Vector4> measurements;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return measurements;
    }
    
    std::string line;
    std::vector<gtsam::Point2> positions; // 先存储所有位置，后续计算速度
    
    // 第一步：读取所有位置数据
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double x, y;
        char comma;
        
        if (iss >> x >> comma >> y) {
            positions.emplace_back(x, y);
        } else {
            iss.clear();
            iss.seekg(0);
            if (iss >> x) {
                positions.emplace_back(x, 0.0);
            }
        }
    }
    file.close();
    
    // 第二步：根据位置差估算初始速度（关键修改）
    double defaultDt = 0.1; // 假设原始数据的默认时间步长（需根据你的数据实际情况调整！）
    for (size_t i = 0; i < positions.size(); ++i) {
        gtsam::Vector4 state;
        state << positions[i].x(), positions[i].y(), 0.0, 0.0;
        
        // 对非第一个点，用前一位置的差分估算速度（避免初始速度为0）
        if (i > 0) {
            double dx = positions[i].x() - positions[i-1].x();
            double dy = positions[i].y() - positions[i-1].y();
            state[2] = dx / defaultDt; // vx = Δx / Δt
            state[3] = dy / defaultDt; // vy = Δy / Δt
        }
        
        measurements.push_back(state);
    }
    
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
    
    for (const auto& key_value : result) {
        gtsam::Key key = key_value.key;
        // 直接使用整数键值，因为我们使用的是整数索引
        gtsam::Vector4 state = result.at<gtsam::Vector4>(key);
        file << state[0] << "," << state[1] << std::endl; // 保存x,y坐标
    }
    
    file.close();
    std::cout << "滤波后的轨迹已保存到: " << filename << std::endl;
}


// 将Values转换为Vector4向量
std::vector<gtsam::Vector4> valuesToVector(const gtsam::Values& result) {
    std::vector<gtsam::Vector4> trajectory;
    
    for (const auto& key_value : result) {
        gtsam::Key key = key_value.key;
        gtsam::Vector4 state = result.at<gtsam::Vector4>(key);
        trajectory.push_back(state);
    }
    
    return trajectory;
}

// 保存Vector4向量到CSV文件（包含时间戳、位置、速度和加速度）
void saveTrajectoryFromVector(const std::vector<gtsam::Vector4>& trajectory, const std::string& filename, double dt) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }
    
    // 创建输出目录
    system("mkdir -p output");
    
    // 写入CSV头
    file << "timestamp,x,y,vx,vy,velocity,acceleration" << std::endl;
    double maxDt = 0.2; // 根据实际场景设置（如10Hz采集对应0.1s，20Hz对应0.05s）
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& state = trajectory[i];
        double timestamp = i * dt; // 计算时间戳
        double x = state[0];
        double y = state[1];
        double vx = state[2];
        double vy = state[3];
        double velocity = std::sqrt(vx * vx + vy * vy);
        
        // 计算加速度（使用中心差分法）
        double acceleration = 0.0;
        if (i > 0 && i < trajectory.size() - 1) {
            const auto& prevState = trajectory[i-1];
            const auto& nextState = trajectory[i+1];
            
            double ax = (nextState[2] - prevState[2]) / (2 * dt);
            double ay = (nextState[3] - prevState[3]) / (2 * dt);
            acceleration = std::sqrt(ax * ax + ay * ay);
        }
        
        file << timestamp << "," << x << "," << y << "," << vx << "," << vy << "," << velocity << "," << acceleration << std::endl;
    }
    
    file.close();
    std::cout << "插值后的轨迹已保存到: " << filename << std::endl;
}


int main() {
    // 1. 读取轨迹数据
    std::vector<gtsam::Vector4> measurements = loadCSV("/home/chen/Documents/GPMPFilter/data/trajectory.csv");
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
    // 自适应时间参数化：根据速度加速度约束计算时间步长
    double maxVelocity = 5.0;      // 最大速度 (m/s)
    double maxAcceleration = 2.0;  // 最大加速度 (m/s²)
    double minDt = 0.005;           // 最小时间步长 (s)
    
    // 计算自适应时间步长
    std::vector<double> adaptiveDts;
    for (size_t i = 0; i < measurements.size() - 1; ++i) {
        const gtsam::Vector4& state1 = measurements[i];
        const gtsam::Vector4& state2 = measurements[i+1];
        
        // 提取位置和速度
        gtsam::Vector2 pos1 = state1.head(2);
        gtsam::Vector2 pos2 = state2.head(2);
        gtsam::Vector2 vel1 = state1.tail(2);
        gtsam::Vector2 vel2 = state2.tail(2);
        
        // 计算位置差
        gtsam::Vector2 deltaPos = pos2 - pos1;
        double distance = deltaPos.norm();
        
        // 计算速度变化
        gtsam::Vector2 deltaVel = vel2 - vel1;
        double deltaVelMagnitude = deltaVel.norm();
        
        // 基于加速度约束计算最小时间
        double timeFromAcceleration = deltaVelMagnitude / maxAcceleration;
        
        // 基于速度约束计算最小时间
        double avgSpeed = (vel1.norm() + vel2.norm()) / 2.0;
        double timeFromDistance = distance / std::max(avgSpeed, 0.1); // 避免除零
        
        // 取两者中的较大值，确保同时满足速度和加速度约束
        double minTime = std::max(timeFromAcceleration, timeFromDistance);
        double maxDt = 0.12; // 根据实际场景设置（60Hz采集对应0.12s）
        double adaptiveDt = std::clamp(minTime, minDt, maxDt); // 限制在 [minDt, maxDt] 之间
        
        adaptiveDts.push_back(adaptiveDt);
    }
    
    // 使用第一个自适应dt作为基准
    const double dt = adaptiveDts[0];
    
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(4) << 0.06, 0.06, 0.2, 0.2).finished());
    
    // 起点/终点约束（先验因子）
    graph.add(gtsam::PriorFactor<gtsam::Vector4>(0, measurements[0], prior_noise));
    graph.add(gtsam::PriorFactor<gtsam::Vector4>(measurements.size()-1,
        measurements.back(), prior_noise));
    
    // GP过程因子（建模动力学连续性）
    auto gp_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(4) << 0.05, 0.05, 0.02, 0.02).finished());
    for (size_t i = 0; i < measurements.size()-1; ++i) {
        // 使用自适应时间步长
        graph.add(std::make_shared<GPFactor>(i, i+1, adaptiveDts[i], gp_noise));
    }
    
    // 测量因子（抑制观测噪声）
    auto meas_noise = gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
    for (size_t i = 1; i < measurements.size()-1; ++i) {
        graph.add(std::make_shared<MeasurementFactor>(
            i, gtsam::Point2(measurements[i][0], measurements[i][1]), meas_noise));
    }
    
    // 速度约束（运动学约束因子）
    auto vel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(2) << 0.1, 0.1).finished());
    for (size_t i = 0; i < measurements.size(); ++i) {
        graph.add(std::make_shared<VelocityConstraint>(i, 5.0, vel_noise));
    }
    
    // 加速度约束（运动学约束因子）
    // 降低噪声协方差，使加速度约束更强
    auto accel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(2) << 0.01, 0.01).finished());
    for (size_t i = 0; i < measurements.size() - 1; ++i) {
        // 加速度约束现在使用自适应时间步长
        graph.add(std::make_shared<AccelerationConstraint>(i, i+1, maxAcceleration, adaptiveDts[i], accel_noise));
    }

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 5. 优化求解
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosity("TERMINATION");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    gtsam::Values result = optimizer.optimize();
    
    // 6. 保存滤波后的轨迹
    saveTrajectory(result, "/home/chen/Documents/GPMPFilter/output/smoothed_trajectory.csv");
    
    // 7. 将优化结果转换为Vector4向量
    std::vector<gtsam::Vector4> smoothedTrajectory = valuesToVector(result);
    
    // 8. 初始化GP插值器
    const int interpolationFactor = 10; // 插值因子，将滤波后的数据点插值为10倍
    // 使用平均自适应时间步长作为插值器的基础时间步长
    double avgAdaptiveDt = 0.0;
    for (double dt_val : adaptiveDts) {
        avgAdaptiveDt += dt_val;
    }
    avgAdaptiveDt /= adaptiveDts.size();
    GPInterpolator interpolator(avgAdaptiveDt, interpolationFactor);

    interpolator.setLengthScale(avgAdaptiveDt * 2);  // 特征长度：建议设为平均时间步的0.5-1倍（值越大越平滑）
    interpolator.setNoiseSigma(0.05);                  // 观测噪声：建议设为0.01-0.05（值越小越贴近原始滤波轨迹）
    
    // 9. 对滤波后的轨迹进行GP插值（仅用于获得平滑轨迹）
    std::vector<gtsam::Vector4> interpolatedTrajectory = interpolator.interpolate(smoothedTrajectory);
    std::cout << "GP插值后得到 " << interpolatedTrajectory.size() << " 个数据点" << std::endl;
    
    // 10. 保存插值后的平滑轨迹（包含时间戳、速度和加速度）
    double interpolatedDt = avgAdaptiveDt / interpolationFactor; // 使用平均自适应时间步长
    saveTrajectoryFromVector(interpolatedTrajectory, "/home/chen/Documents/GPMPFilter/output/interpolated_trajectory.csv", interpolatedDt);
    
    
    // 12. 初始化时间参数化器
    double defaultDt = 0.01;       // 默认时间步长 (s)
    
    // 14. 计算并输出速度和加速度统计信息
    double maxVel = 0.0, avgVel = 0.0;
    double maxAccel = 0.0, avgAccel = 0.0;
    int count = 0;
    

    // 直接使用GP插值结果计算统计信息
    for (size_t i = 0; i < interpolatedTrajectory.size(); ++i) {
        const auto& state = interpolatedTrajectory[i];
        double vel = std::sqrt(state[2] * state[2] + state[3] * state[3]);
        maxVel = std::max(maxVel, vel);
        avgVel += vel;
        
        // 计算加速度（使用中心差分法）
        if (i > 0 && i < interpolatedTrajectory.size() - 1) {
            const auto& prevState = interpolatedTrajectory[i-1];
            const auto& nextState = interpolatedTrajectory[i+1];
            
            double currentInterpolatedDt = avgAdaptiveDt / interpolationFactor; // 使用平均自适应时间步长
            double ax = (nextState[2] - prevState[2]) / (2 * currentInterpolatedDt);
            double ay = (nextState[3] - prevState[3]) / (2 * currentInterpolatedDt);
            double accel = std::sqrt(ax * ax + ay * ay);
            maxAccel = std::max(maxAccel, accel);
            avgAccel += accel;
            count++;
        }
    }
    
    avgVel /= interpolatedTrajectory.size();
    if (count > 0) {
        avgAccel /= count;
    }
    
    std::cout << "=== GP插值结果统计信息 ===" << std::endl;
    std::cout << "最大速度: " << maxVel << " m/s" << std::endl;
    std::cout << "平均速度: " << avgVel << " m/s" << std::endl;
    std::cout << "最大加速度: " << maxAccel << " m/s²" << std::endl;
    std::cout << "平均加速度: " << avgAccel << " m/s²" << std::endl;
    
    // 输出自适应时间步长统计信息
    std::cout << "=== 自适应时间步长统计信息 ===" << std::endl;
    double minAdaptiveDt = *std::min_element(adaptiveDts.begin(), adaptiveDts.end());
    double maxAdaptiveDt = *std::max_element(adaptiveDts.begin(), adaptiveDts.end());
    std::cout << "最小自适应时间步长: " << minAdaptiveDt << " s" << std::endl;
    std::cout << "最大自适应时间步长: " << maxAdaptiveDt << " s" << std::endl;
    std::cout << "平均自适应时间步长: " << avgAdaptiveDt << " s" << std::endl;
    std::cout << "原始固定时间步长: " << 0.1 << " s" << std::endl;
    
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