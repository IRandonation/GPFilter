// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/base/Vector.h>
#include "GPFactor.h"
#include "MeasurementFactor.h"
#include "VelocityConstraint.h"
#include "AccelerationConstraint.h"

// 加载CSV数据文件
std::vector<gtsam::Vector4> loadCSV(const std::string& filename) {
    std::vector<gtsam::Vector4> measurements;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return measurements;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double value;
        if (iss >> value) {
            // 将位置数据转换为Vector4格式 [x, y, vx, vy]
            // 由于数据只有一维，我们将其作为x坐标，y设为0
            gtsam::Vector4 state;
            state << value, 0.0, 0.0, 0.0; // 初始速度设为0
            measurements.push_back(state);
        }
    }
    
    file.close();
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
        file << state[0] << std::endl; // 只保存x坐标
    }
    
    file.close();
    std::cout << "滤波后的轨迹已保存到: " << filename << std::endl;
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
    const double dt = 0.1;
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(4) << 0.1, 0.1, 0.5, 0.5).finished());
    
    // 起点/终点约束（先验因子）
    graph.add(gtsam::PriorFactor<gtsam::Vector4>(0, measurements[0], prior_noise));
    graph.add(gtsam::PriorFactor<gtsam::Vector4>(measurements.size()-1,
        measurements.back(), prior_noise));
    
    // GP过程因子（建模动力学连续性）
    auto gp_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(4) << 0.05, 0.05, 0.1, 0.1).finished());
    for (size_t i = 0; i < measurements.size()-1; ++i) {
        graph.add(std::make_shared<GPFactor>(i, i+1, dt, gp_noise));
    }
    
    // 测量因子（抑制观测噪声）
    auto meas_noise = gtsam::noiseModel::Isotropic::Sigma(2, 0.3);
    for (size_t i = 1; i < measurements.size()-1; ++i) {
        graph.add(std::make_shared<MeasurementFactor>(
            i, gtsam::Point2(measurements[i][0], measurements[i][1]), meas_noise));
    }
    
    // 速度约束（运动学约束因子）
    auto vel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(2) << 0.1, 0.1).finished());
    for (size_t i = 0; i < measurements.size(); ++i) {
        graph.add(std::make_shared<VelocityConstraint>(i, 2.0, vel_noise));
    }
    
    // 加速度约束（运动学约束因子）
    auto accel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(2) << 0.1, 0.1).finished());
    for (size_t i = 0; i < measurements.size(); ++i) {
        graph.add(std::make_shared<AccelerationConstraint>(i, 1.0, accel_noise));
    }
    
    // 5. 优化求解
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosity("TERMINATION");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    gtsam::Values result = optimizer.optimize();
    
    // 6. 保存结果
    saveTrajectory(result, "/home/chen/Documents/GPMPFilter/output/smoothed_trajectory.csv");
    
    // 7. 输出一些统计信息
    std::cout << "优化完成！" << std::endl;
    std::cout << "初始误差: " << graph.error(initial) << std::endl;
    std::cout << "最终误差: " << graph.error(result) << std::endl;
    
    return 0;
}