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
#include "GPFactor.h"
#include "MeasurementFactor.h"
#include "VelocityConstraint.h"
#include "AccelerationConstraint.h"
#include "GPInterpolator.h"
#include "VelocityEstimator.h"

// 加载CSV数据文件
std::vector<Vector12> loadCSV(const std::string& filename) {
    std::vector<Vector12> measurements;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return measurements;
    }
    
    std::string line;
    std::vector<gtsam::Pose3> poses; // 先存储所有位姿，后续计算速度
    
    // 第一步：读取所有位姿数据
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
        
        // 根据数据维度创建位姿
        if (values.size() >= 6) {
            // 6自由度数据：x, y, z, roll, pitch, yaw
            gtsam::Point3 position(values[0], values[1], values[2]);
            gtsam::Rot3 rotation = gtsam::Rot3::Yaw(values[5]) *
                                  gtsam::Rot3::Pitch(values[4]) *
                                  gtsam::Rot3::Roll(values[3]);
            poses.emplace_back(rotation, position);
        } else if (values.size() >= 3) {
            // 3自由度数据：x, y, z
            gtsam::Point3 position(values[0], values[1], values[2]);
            gtsam::Rot3 rotation = gtsam::Rot3::Identity(); // 无旋转
            poses.emplace_back(rotation, position);
        } else if (values.size() >= 2) {
            // 2自由度数据：x, y
            gtsam::Point3 position(values[0], values[1], 0.0); // z设为0
            gtsam::Rot3 rotation = gtsam::Rot3::Identity(); // 无旋转
            poses.emplace_back(rotation, position);
        } else if (values.size() >= 1) {
            // 1自由度数据：x
            gtsam::Point3 position(values[0], 0.0, 0.0); // y, z设为0
            gtsam::Rot3 rotation = gtsam::Rot3::Identity(); // 无旋转
            poses.emplace_back(rotation, position);
        }
    }
    file.close();
    
    // 第二步：使用速度估计器估算初始速度（现在包含线速度和角速度）
    double initialDt = 0.1; // 假设原始数据的默认时间步长（需根据你的数据实际情况调整！）
    
    // 创建速度估计器，使用匀速模型方法
    // 可以选择以下方法：
    // VelocityEstimator::SIMPLE_DIFFERENCE - 简单差分法（原始方法）
    // VelocityEstimator::CONSTANT_VELOCITY - 匀速模型估计（推荐）
    // VelocityEstimator::KALMAN_FILTER - 卡尔曼滤波估计
    VelocityEstimator estimator(VelocityEstimator::CONSTANT_VELOCITY, 5, initialDt);
    
    // 使用速度估计器估算速度（现在返回12维状态向量）
    measurements = estimator.estimateVelocities(poses);
    
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
    
    std::cout << "检测到 " << poses.size() << " 个位姿点，维度: ";
    if (poses.size() > 0) {
        if (poses[0].rotation().equals(gtsam::Rot3::Identity())) {
            if (poses[0].z() == 0.0) {
                if (poses[0].y() == 0.0) {
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
    file << "x,y,z,roll,pitch,yaw,vx,vy,vz,vroll,vpitch,vyaw" << std::endl;
    
    for (const auto& key_value : result) {
        gtsam::Key key = key_value.key;
        // 直接使用整数键值，因为我们使用的是整数索引
        Vector12 state = result.at<Vector12>(key);
        // 现在state包含完整的12维状态信息
        file << state[0] << "," << state[1] << "," << state[2] << ","  // x, y, z
             << state[3] << "," << state[4] << "," << state[5] << ","  // roll, pitch, yaw
             << state[6] << "," << state[7] << "," << state[8] << ","  // vx, vy, vz
             << state[9] << "," << state[10] << "," << state[11] << std::endl; // vroll, vpitch, vyaw
    }
    
    file.close();
    std::cout << "滤波后的轨迹已保存到: " << filename << std::endl;
}


// 将Values转换为Vector12向量
std::vector<Vector12> valuesToVector(const gtsam::Values& result) {
    std::vector<Vector12> trajectory;
    
    for (const auto& key_value : result) {
        gtsam::Key key = key_value.key;
        Vector12 state = result.at<Vector12>(key);
        trajectory.push_back(state);
    }
    
    return trajectory;
}

// 保存Vector12向量到CSV文件（包含时间戳、位置、姿态、速度和加速度）
void saveTrajectoryFromVector(const std::vector<Vector12>& trajectory, const std::string& filename, double dt) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }
    
    // 创建输出目录
    system("mkdir -p output");
    
    // 写入CSV头
    file << "timestamp,x,y,z,roll,pitch,yaw,vx,vy,vz,vroll,vpitch,vyaw,linear_velocity,angular_velocity,linear_acceleration,angular_acceleration" << std::endl;
    
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& state = trajectory[i];
        double timestamp = i * dt; // 计算时间戳
        
        // 位置和姿态
        double x = state[0];
        double y = state[1];
        double z = state[2];
        double roll = state[3];
        double pitch = state[4];
        double yaw = state[5];
        
        // 线速度和角速度
        double vx = state[6];
        double vy = state[7];
        double vz = state[8];
        double vroll = state[9];
        double vpitch = state[10];
        double vyaw = state[11];
        
        // 计算线速度和角速度的大小
        double linear_velocity = std::sqrt(vx * vx + vy * vy + vz * vz);
        double angular_velocity = std::sqrt(vroll * vroll + vpitch * vpitch + vyaw * vyaw);
        
        // 计算线加速度和角加速度（使用中心差分法）
        double linear_acceleration = 0.0;
        double angular_acceleration = 0.0;
        
        if (i > 0 && i < trajectory.size() - 1) {
            const auto& prevState = trajectory[i-1];
            const auto& nextState = trajectory[i+1];
            
            // 线加速度
            double ax = (nextState[6] - prevState[6]) / (2 * dt);
            double ay = (nextState[7] - prevState[7]) / (2 * dt);
            double az = (nextState[8] - prevState[8]) / (2 * dt);
            linear_acceleration = std::sqrt(ax * ax + ay * ay + az * az);
            
            // 角加速度
            double aroll = (nextState[9] - prevState[9]) / (2 * dt);
            double apitch = (nextState[10] - prevState[10]) / (2 * dt);
            double ayaw = (nextState[11] - prevState[11]) / (2 * dt);
            angular_acceleration = std::sqrt(aroll * aroll + apitch * apitch + ayaw * ayaw);
        }
        
        file << timestamp << ","
             << x << "," << y << "," << z << ","
             << roll << "," << pitch << "," << yaw << ","
             << vx << "," << vy << "," << vz << ","
             << vroll << "," << vpitch << "," << vyaw << ","
             << linear_velocity << "," << angular_velocity << ","
             << linear_acceleration << "," << angular_acceleration << std::endl;
    }
    
    file.close();
    std::cout << "插值后的轨迹已保存到: " << filename << std::endl;
}


int main() {
    // 1. 读取轨迹数据
    std::vector<Vector12> measurements = loadCSV("/home/chen/Documents/GPMPFilter/data/trajectory.csv");
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
    // 首先计算初始线速度和角速度的最大值，以设置合理的约束
    double maxInitialLinearVel = 0.0;
    double maxInitialAngularVel = 0.0;
    for (const auto& state : measurements) {
        // 线速度
        double linear_vel = std::sqrt(state[6] * state[6] + state[7] * state[7] + state[8] * state[8]);
        maxInitialLinearVel = std::max(maxInitialLinearVel, linear_vel);
        
        // 角速度
        double angular_vel = std::sqrt(state[9] * state[9] + state[10] * state[10] + state[11] * state[11]);
        maxInitialAngularVel = std::max(maxInitialAngularVel, angular_vel);
    }
    
    // 设置最大线速度、角速度和加速度约束，基于初始估计值
    double maxLinearVelocity = std::max(20.0, maxInitialLinearVel * 2.0);      // 最大线速度 (m/s)
    double maxAngularVelocity = std::max(3.14, maxInitialAngularVel * 2.0);   // 最大角速度 (rad/s)
    double maxLinearAcceleration = 5.0;  // 最大线加速度 (m/s²)
    double maxAngularAcceleration = 3.14; // 最大角加速度 (rad/s²)
    
    std::cout << "初始最大线速度估计: " << maxInitialLinearVel << " m/s" << std::endl;
    std::cout << "初始最大角速度估计: " << maxInitialAngularVel << " rad/s" << std::endl;
    std::cout << "设置最大线速度约束: " << maxLinearVelocity << " m/s" << std::endl;
    std::cout << "设置最大角速度约束: " << maxAngularVelocity << " rad/s" << std::endl;
    std::cout << "设置最大线加速度约束: " << maxLinearAcceleration << " m/s²" << std::endl;
    std::cout << "设置最大角加速度约束: " << maxAngularAcceleration << " rad/s²" << std::endl;
    
    // 调整噪声模型参数，使其更适合新的速度估计
    // 先验因子噪声：位置和姿态更精确，速度较宽松
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(12) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).finished());  // 位置姿态噪声小，速度噪声大
    
    // 起点/终点约束（先验因子）
    graph.add(gtsam::PriorFactor<Vector12>(0, measurements[0], prior_noise));
    graph.add(gtsam::PriorFactor<Vector12>(measurements.size()-1, measurements.back(), prior_noise));
    
    // GP过程因子（建模动力学连续性）
    auto gp_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(12) << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5).finished());  // 位置姿态约束中等，速度约束宽松
    for (size_t i = 0; i < measurements.size()-1; ++i) {
        // 使用固定时间步长
        graph.add(std::make_shared<GPFactor>(i, i+1, dt, gp_noise));
    }
    
    // 测量因子（抑制观测噪声）
    auto meas_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1.5);  // 6维测量噪声（位置和姿态）
    for (size_t i = 1; i < measurements.size()-1; ++i) {
        gtsam::Point3 position(measurements[i][0], measurements[i][1], measurements[i][2]);
        gtsam::Rot3 rotation = gtsam::Rot3::Yaw(measurements[i][5]) *
                              gtsam::Rot3::Pitch(measurements[i][4]) *
                              gtsam::Rot3::Roll(measurements[i][3]);
        gtsam::Pose3 pose(rotation, position);
        graph.add(std::make_shared<MeasurementFactor>(i, pose, meas_noise));
    }
    
    // 速度约束（运动学约束因子）
    auto vel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 2.0, 2.0, 2.0, 2.0, 2.0, 2.0).finished());  // 6维速度约束噪声（线速度和角速度）
    for (size_t i = 0; i < measurements.size(); ++i) {
        graph.add(std::make_shared<VelocityConstraint>(i, maxLinearVelocity, maxAngularVelocity, vel_noise));
    }
    
    // 加速度约束（运动学约束因子）
    auto accel_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 2.0, 2.0, 2.0, 2.0, 2.0, 2.0).finished());  // 6维加速度约束噪声（线加速度和角加速度）
    for (size_t i = 0; i < measurements.size() - 1; ++i) {
        // 加速度约束使用固定时间步长
        graph.add(std::make_shared<AccelerationConstraint>(i, i+1, maxLinearAcceleration, maxAngularAcceleration, dt, accel_noise));
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
    std::vector<Vector12> smoothedTrajectory = valuesToVector(result);
    
    // 8. 初始化GP插值器
    const int interpolationFactor = 10; // 插值因子，将滤波后的数据点插值为10倍
    
    // 使用固定时间步长作为插值器的基础时间步长
    GPInterpolator interpolator(dt, interpolationFactor);

    interpolator.setLengthScale(dt * 1.10);  // 特征长度：设为时间步的1倍，平衡平滑性与跟踪性
    interpolator.setNoiseSigma(0.25);      // 观测噪声：减小噪声，使插值更贴近原始滤波轨迹
    
    // 9. 对滤波后的轨迹进行GP插值（仅用于获得平滑轨迹）
    std::vector<Vector12> interpolatedTrajectory = interpolator.interpolate(smoothedTrajectory);
    std::cout << "GP插值后得到 " << interpolatedTrajectory.size() << " 个数据点" << std::endl;
    
    // 10. 保存插值后的平滑轨迹（包含时间戳、速度和加速度）
    double interpolatedDt = dt / interpolationFactor; // 使用固定时间步长
    saveTrajectoryFromVector(interpolatedTrajectory, "/home/chen/Documents/GPMPFilter/output/interpolated_trajectory.csv", interpolatedDt);
    
    
    // 12. 初始化时间参数化器
    // 使用固定时间步长作为基准
    
    // 14. 计算并输出速度和加速度统计信息
    double maxLinearVel = 0.0, avgLinearVel = 0.0;
    double maxAngularVel = 0.0, avgAngularVel = 0.0;
    double maxLinearAccel = 0.0, avgLinearAccel = 0.0;
    double maxAngularAccel = 0.0, avgAngularAccel = 0.0;
    int count = 0;
    

    // 直接使用GP插值结果计算统计信息
    for (size_t i = 0; i < interpolatedTrajectory.size(); ++i) {
        const auto& state = interpolatedTrajectory[i];
        
        // 线速度统计
        double linear_vel = std::sqrt(state[6] * state[6] + state[7] * state[7] + state[8] * state[8]);
        maxLinearVel = std::max(maxLinearVel, linear_vel);
        avgLinearVel += linear_vel;
        
        // 角速度统计
        double angular_vel = std::sqrt(state[9] * state[9] + state[10] * state[10] + state[11] * state[11]);
        maxAngularVel = std::max(maxAngularVel, angular_vel);
        avgAngularVel += angular_vel;
        
        // 计算加速度（使用中心差分法）
        if (i > 0 && i < interpolatedTrajectory.size() - 1) {
            const auto& prevState = interpolatedTrajectory[i-1];
            const auto& nextState = interpolatedTrajectory[i+1];
            
            // 使用插值后的时间步长
            double currentInterpolatedDt = dt / interpolationFactor;
            
            // 线加速度
            double ax = (nextState[6] - prevState[6]) / (2 * currentInterpolatedDt);
            double ay = (nextState[7] - prevState[7]) / (2 * currentInterpolatedDt);
            double az = (nextState[8] - prevState[8]) / (2 * currentInterpolatedDt);
            double linear_accel = std::sqrt(ax * ax + ay * ay + az * az);
            maxLinearAccel = std::max(maxLinearAccel, linear_accel);
            avgLinearAccel += linear_accel;
            
            // 角加速度
            double aroll = (nextState[9] - prevState[9]) / (2 * currentInterpolatedDt);
            double apitch = (nextState[10] - prevState[10]) / (2 * currentInterpolatedDt);
            double ayaw = (nextState[11] - prevState[11]) / (2 * currentInterpolatedDt);
            double angular_accel = std::sqrt(aroll * aroll + apitch * apitch + ayaw * ayaw);
            maxAngularAccel = std::max(maxAngularAccel, angular_accel);
            avgAngularAccel += angular_accel;
            
            count++;
        }
    }
    
    avgLinearVel /= interpolatedTrajectory.size();
    avgAngularVel /= interpolatedTrajectory.size();
    if (count > 0) {
        avgLinearAccel /= count;
        avgAngularAccel /= count;
    }

    
    std::cout << "=== GP插值结果统计信息 ===" << std::endl;
    std::cout << "最大线速度: " << maxLinearVel << " m/s" << std::endl;
    std::cout << "平均线速度: " << avgLinearVel << " m/s" << std::endl;
    std::cout << "最大角速度: " << maxAngularVel << " rad/s" << std::endl;
    std::cout << "平均角速度: " << avgAngularVel << " rad/s" << std::endl;
    std::cout << "最大线加速度: " << maxLinearAccel << " m/s²" << std::endl;
    std::cout << "平均线加速度: " << avgLinearAccel << " m/s²" << std::endl;
    std::cout << "最大角加速度: " << maxAngularAccel << " rad/s²" << std::endl;
    std::cout << "平均角加速度: " << avgAngularAccel << " rad/s²" << std::endl;
    
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