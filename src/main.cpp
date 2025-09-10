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
#include "VelocityEstimator.h"

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
    
    // 第二步：使用速度估计器估算初始速度
    double initialDt = 0.1; // 假设原始数据的默认时间步长（需根据你的数据实际情况调整！）
    
    // 创建速度估计器，使用匀速模型方法
    // 可以选择以下方法：
    // VelocityEstimator::SIMPLE_DIFFERENCE - 简单差分法（原始方法）
    // VelocityEstimator::CONSTANT_VELOCITY - 匀速模型估计（推荐）
    // VelocityEstimator::KALMAN_FILTER - 卡尔曼滤波估计
    VelocityEstimator estimator(VelocityEstimator::CONSTANT_VELOCITY, 5, initialDt);
    
    // 使用速度估计器估算速度
    measurements = estimator.estimateVelocities(positions);
    
    std::cout << "使用速度估计器完成速度估计，方法: ";
    switch (estimator.getEstimationMethod()) {
        case VelocityEstimator::SIMPLE_DIFFERENCE:
            std::cout << "简单差分法";
            break;
        case VelocityEstimator::CONSTANT_VELOCITY:
            std::cout << "匀速模型估计";
            break;
        case VelocityEstimator::KALMAN_FILTER:
            std::cout << "卡尔曼滤波估计";
            break;
    }
    std::cout << ", 窗口大小: " << estimator.getWindowSize() << std::endl;
    
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
    // 使用固定时间步长
    const double dt = 0.1;  // 固定时间步长 (s)
    
    // 根据新的速度估计方法调整约束参数
    // 首先计算初始速度的最大值，以设置合理的约束
    double maxInitialVel = 0.0;
    for (const auto& state : measurements) {
        double vel = std::sqrt(state[2] * state[2] + state[3] * state[3]);
        maxInitialVel = std::max(maxInitialVel, vel);
    }
    
    // 设置最大速度和加速度约束，基于初始估计值
    double maxVelocity = std::max(20.0, maxInitialVel * 2.0);      // 最大速度 (m/s) - 放宽约束
    double maxAcceleration = 5.0;  // 最大加速度 (m/s²) - 放宽约束
    
    std::cout << "初始最大速度估计: " << maxInitialVel << " m/s" << std::endl;
    std::cout << "设置最大速度约束: " << maxVelocity << " m/s" << std::endl;
    std::cout << "设置最大加速度约束: " << maxAcceleration << " m/s²" << std::endl;
    
    // 调整噪声模型参数，使其更适合新的速度估计
    // 先验因子噪声：位置更精确，速度较宽松
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(4) << 0.1, 0.1, 1.0, 1.0).finished());  // 放宽位置和速度噪声
    
    // 起点/终点约束（先验因子）
    graph.add(gtsam::PriorFactor<gtsam::Vector4>(0, measurements[0], prior_noise));
    graph.add(gtsam::PriorFactor<gtsam::Vector4>(measurements.size()-1,
        measurements.back(), prior_noise));
    
    // GP过程因子（建模动力学连续性）
    auto gp_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(4) << 0.5, 0.5, 1.0, 1.0).finished());  // 进一步放宽GP约束
    for (size_t i = 0; i < measurements.size()-1; ++i) {
        // 使用固定时间步长
        graph.add(std::make_shared<GPFactor>(i, i+1, dt, gp_noise));
    }
    
    // 测量因子（抑制观测噪声）
    auto meas_noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.5);  // 大幅放宽测量噪声
    for (size_t i = 1; i < measurements.size()-1; ++i) {
        graph.add(std::make_shared<MeasurementFactor>(
            i, gtsam::Point2(measurements[i][0], measurements[i][1]), meas_noise));
    }
    
    // 速度约束（运动学约束因子）
    auto vel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(2) << 3.0, 3.0).finished());  // 放宽速度约束噪声
    for (size_t i = 0; i < measurements.size(); ++i) {
        graph.add(std::make_shared<VelocityConstraint>(i, maxVelocity, vel_noise));
    }
    
    // 加速度约束（运动学约束因子）
    auto accel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(2) << 3.0, 3.0).finished());  // 放宽加速度约束噪声
    for (size_t i = 0; i < measurements.size() - 1; ++i) {
        // 加速度约束使用固定时间步长
        graph.add(std::make_shared<AccelerationConstraint>(i, i+1, maxAcceleration, dt, accel_noise));
    }

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 输出初始误差信息
    std::cout << "=== 初始误差分析 ===" << std::endl;
    std::cout << "初始总误差: " << graph.error(initial) << std::endl;
    
    // 检查各个因子的贡献
    for (size_t i = 0; i < graph.size(); ++i) {
        double factor_error = graph.at(i)->error(initial);
        std::cout << "因子 " << i << " 误差: " << factor_error << std::endl;
    }
    
    // 5. 优化求解
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosity("TERMINATION");
    params.setMaxIterations(1000);  // 增加最大迭代次数
    params.setRelativeErrorTol(1e-5);  // 设置相对误差容差
    params.setAbsoluteErrorTol(1e-3);  // 设置绝对误差容差
    params.setlambdaInitial(1e-5);  // 设置较小的初始lambda值
    params.setlambdaFactor(2.0);  // 设置lambda增长因子
    params.setlambdaUpperBound(1e10);  // 设置lambda上界
    params.setlambdaLowerBound(1e-10);  // 设置lambda下界
    params.setUseFixedLambdaFactor(false);  // 允许动态调整lambda
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    gtsam::Values result = optimizer.optimize();
    
    // 6. 保存滤波后的轨迹
    saveTrajectory(result, "/home/chen/Documents/GPMPFilter/output/smoothed_trajectory.csv");
    
    // 7. 将优化结果转换为Vector4向量
    std::vector<gtsam::Vector4> smoothedTrajectory = valuesToVector(result);
    
    // 8. 初始化GP插值器
    const int interpolationFactor = 10; // 插值因子，将滤波后的数据点插值为10倍
    
    // 使用固定时间步长作为插值器的基础时间步长
    GPInterpolator interpolator(dt, interpolationFactor);

    interpolator.setLengthScale(dt * 1.10);  // 特征长度：设为时间步的1倍，平衡平滑性与跟踪性
    interpolator.setNoiseSigma(0.25);      // 观测噪声：减小噪声，使插值更贴近原始滤波轨迹
    
    // 9. 对滤波后的轨迹进行GP插值（仅用于获得平滑轨迹）
    std::vector<gtsam::Vector4> interpolatedTrajectory = interpolator.interpolate(smoothedTrajectory);
    std::cout << "GP插值后得到 " << interpolatedTrajectory.size() << " 个数据点" << std::endl;
    
    // 10. 保存插值后的平滑轨迹（包含时间戳、速度和加速度）
    double interpolatedDt = dt / interpolationFactor; // 使用固定时间步长
    saveTrajectoryFromVector(interpolatedTrajectory, "/home/chen/Documents/GPMPFilter/output/interpolated_trajectory.csv", interpolatedDt);
    
    
    // 12. 初始化时间参数化器
    // 使用固定时间步长作为基准
    
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
            
            // 使用插值后的时间步长
            double currentInterpolatedDt = dt / interpolationFactor;
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

    for (size_t i = 0; i < 10; ++i) {
        const auto& state_before = initial.at<gtsam::Vector4>(i);
        const auto& state_after = result.at<gtsam::Vector4>(i);
        std::cout << "点 " << i << ": 优化前 = " << state_before.transpose()
              << ", 优化后 = " << state_after.transpose() << std::endl;
    }
    
    std::cout << "=== GP插值结果统计信息 ===" << std::endl;
    std::cout << "最大速度: " << maxVel << " m/s" << std::endl;
    std::cout << "平均速度: " << avgVel << " m/s" << std::endl;
    std::cout << "最大加速度: " << maxAccel << " m/s²" << std::endl;
    std::cout << "平均加速度: " << avgAccel << " m/s²" << std::endl;
    
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