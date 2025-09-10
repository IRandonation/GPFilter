// src/VelocityEstimator.cpp
#include "VelocityEstimator.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

VelocityEstimator::VelocityEstimator(EstimationMethod method, int windowSize, double dt)
    : method_(method), windowSize_(windowSize), dt_(dt) {
    // 确保窗口大小至少为3
    windowSize_ = std::max(3, windowSize_);
    // 确保窗口大小为奇数，便于中心化处理
    if (windowSize_ % 2 == 0) {
        windowSize_++;
    }
}

std::vector<Vector12> VelocityEstimator::estimateVelocities(
    const std::vector<gtsam::Pose3>& poses) {
    
    switch (method_) {
        case SIMPLE_DIFFERENCE:
            return estimateWithSimpleDifference(poses);
        case CONSTANT_VELOCITY:
            return estimateWithConstantVelocity(poses);
        case KALMAN_FILTER:
            return estimateWithKalmanFilter(poses);
        default:
            return estimateWithConstantVelocity(poses); // 默认使用匀速模型
    }
}

void VelocityEstimator::setEstimationMethod(EstimationMethod method) {
    method_ = method;
}

void VelocityEstimator::setWindowSize(int windowSize) {
    windowSize_ = std::max(3, windowSize);
    if (windowSize_ % 2 == 0) {
        windowSize_++;
    }
}

void VelocityEstimator::setTimeStep(double dt) {
    dt_ = dt;
}

VelocityEstimator::EstimationMethod VelocityEstimator::getEstimationMethod() const {
    return method_;
}

int VelocityEstimator::getWindowSize() const {
    return windowSize_;
}

double VelocityEstimator::getTimeStep() const {
    return dt_;
}

std::vector<Vector12> VelocityEstimator::estimateWithSimpleDifference(
    const std::vector<gtsam::Pose3>& poses) {
    
    std::vector<Vector12> states;
    
    for (size_t i = 0; i < poses.size(); ++i) {
        Vector12 state = Vector12::Zero(12);
        // 设置位置和姿态
        state[0] = poses[i].x();
        state[1] = poses[i].y();
        state[2] = poses[i].z();
        
        // 提取姿态（roll, pitch, yaw）
        gtsam::Rot3 rot = poses[i].rotation();
        gtsam::Vector3 rpy = rot.rpy();
        state[3] = rpy[0]; // roll
        state[4] = rpy[1]; // pitch
        state[5] = rpy[2]; // yaw
        
        // 对非第一个点，用前一位置的差分估算速度
        if (i > 0) {
            // 计算线速度
            double dx = poses[i].x() - poses[i-1].x();
            double dy = poses[i].y() - poses[i-1].y();
            double dz = poses[i].z() - poses[i-1].z();
            
            // 计算角速度
            gtsam::Rot3 prevRot = poses[i-1].rotation();
            gtsam::Rot3 currRot = poses[i].rotation();
            gtsam::Rot3 deltaRot = prevRot.inverse() * currRot;
            gtsam::Vector3 deltaAngles = deltaRot.rpy();
            
            // 设置线速度
            state[6] = dx / dt_; // vx = Δx / Δt
            state[7] = dy / dt_; // vy = Δy / Δt
            state[8] = dz / dt_; // vz = Δz / Δt
            
            // 设置角速度
            state[9] = deltaAngles[0] / dt_; // vroll = Δroll / Δt
            state[10] = deltaAngles[1] / dt_; // vpitch = Δpitch / Δt
            state[11] = deltaAngles[2] / dt_; // vyaw = Δyaw / Δt
        }
        
        states.push_back(state);
    }
    
    return states;
}

std::vector<Vector12> VelocityEstimator::estimateWithConstantVelocity(
    const std::vector<gtsam::Pose3>& poses) {
    
    std::vector<Vector12> states;
    int halfWindow = windowSize_ / 2;
    
    for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
        Vector12 state = Vector12::Zero(12);
        // 设置位置和姿态
        state[0] = poses[i].x();
        state[1] = poses[i].y();
        state[2] = poses[i].z();
        
        // 提取姿态（roll, pitch, yaw）
        gtsam::Rot3 rot = poses[i].rotation();
        gtsam::Vector3 rpy = rot.rpy();
        state[3] = rpy[0]; // roll
        state[4] = rpy[1]; // pitch
        state[5] = rpy[2]; // yaw
        
        // 确定局部窗口的起始和结束索引
        int startIdx = std::max(0, i - halfWindow);
        int endIdx = std::min(static_cast<int>(poses.size()) - 1, i + halfWindow);
        
        // 确保窗口内有足够的点
        if (endIdx - startIdx >= 2) {
            // 使用最小二乘法拟合局部匀速模型
            Vector12 velocity = fitLocalConstantVelocity(poses, startIdx, endIdx);
            state[6] = velocity[0]; // vx
            state[7] = velocity[1]; // vy
            state[8] = velocity[2]; // vz
            state[9] = velocity[3]; // vroll
            state[10] = velocity[4]; // vpitch
            state[11] = velocity[5]; // vyaw
        } else if (i > 0) {
            // 如果窗口内点数不足，回退到简单差分法
            // 计算线速度
            double dx = poses[i].x() - poses[i-1].x();
            double dy = poses[i].y() - poses[i-1].y();
            double dz = poses[i].z() - poses[i-1].z();
            
            // 计算角速度
            gtsam::Rot3 prevRot = poses[i-1].rotation();
            gtsam::Rot3 currRot = poses[i].rotation();
            gtsam::Rot3 deltaRot = prevRot.inverse() * currRot;
            gtsam::Vector3 deltaAngles = deltaRot.rpy();
            
            // 设置线速度
            state[6] = dx / dt_; // vx = Δx / Δt
            state[7] = dy / dt_; // vy = Δy / Δt
            state[8] = dz / dt_; // vz = Δz / Δt
            
            // 设置角速度
            state[9] = deltaAngles[0] / dt_; // vroll = Δroll / Δt
            state[10] = deltaAngles[1] / dt_; // vpitch = Δpitch / Δt
            state[11] = deltaAngles[2] / dt_; // vyaw = Δyaw / Δt
        }
        
        states.push_back(state);
    }
    
    return states;
}

std::vector<Vector12> VelocityEstimator::estimateWithKalmanFilter(
    const std::vector<gtsam::Pose3>& poses) {
    
    // 状态向量: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
    // 观测向量: [x, y, z, roll, pitch, yaw]
    
    std::vector<Vector12> states;
    
    // 初始化卡尔曼滤波参数
    Vector12 x = Vector12::Zero(12); // 状态向量
    // 设置初始位置
    x[0] = poses[0].x();
    x[1] = poses[0].y();
    x[2] = poses[0].z();
    // 设置初始姿态
    gtsam::Rot3 initRot = poses[0].rotation();
    gtsam::Vector3 initRPY = initRot.rpy();
    x[3] = initRPY[0]; // roll
    x[4] = initRPY[1]; // pitch
    x[5] = initRPY[2]; // yaw
    // 速度初始为0
    
    // 状态转移矩阵 (匀速模型) - 12x12
    gtsam::Matrix F = gtsam::Matrix::Zero(12, 12);
    // 位置部分
    F(0, 0) = 1; F(0, 6) = dt_;  // x = x + vx * dt
    F(1, 1) = 1; F(1, 7) = dt_;  // y = y + vy * dt
    F(2, 2) = 1; F(2, 8) = dt_;  // z = z + vz * dt
    // 姿态部分
    F(3, 3) = 1; F(3, 9) = dt_;  // roll = roll + vroll * dt
    F(4, 4) = 1; F(4, 10) = dt_; // pitch = pitch + vpitch * dt
    F(5, 5) = 1; F(5, 11) = dt_; // yaw = yaw + vyaw * dt
    // 速度部分（保持不变）
    F(6, 6) = 1; // vx
    F(7, 7) = 1; // vy
    F(8, 8) = 1; // vz
    F(9, 9) = 1; // vroll
    F(10, 10) = 1; // vpitch
    F(11, 11) = 1; // vyaw
    
    // 观测矩阵 - 6x12
    gtsam::Matrix H = gtsam::Matrix::Zero(6, 12);
    // 观测位置
    H(0, 0) = 1; // x
    H(1, 1) = 1; // y
    H(2, 2) = 1; // z
    // 观测姿态
    H(3, 3) = 1; // roll
    H(4, 4) = 1; // pitch
    H(5, 5) = 1; // yaw
    
    // 过程噪声协方差矩阵
    double q_pos = 0.01;      // 位置过程噪声
    double q_rot = 0.01;      // 姿态过程噪声
    double q_vel = 0.1;       // 速度过程噪声
    double q_angular_vel = 0.1; // 角速度过程噪声
    gtsam::Matrix Q = gtsam::Matrix::Zero(12, 12);
    Q(0, 0) = q_pos;      Q(1, 1) = q_pos;      Q(2, 2) = q_pos;      // 位置
    Q(3, 3) = q_rot;      Q(4, 4) = q_rot;      Q(5, 5) = q_rot;      // 姿态
    Q(6, 6) = q_vel;      Q(7, 7) = q_vel;      Q(8, 8) = q_vel;      // 线速度
    Q(9, 9) = q_angular_vel; Q(10, 10) = q_angular_vel; Q(11, 11) = q_angular_vel; // 角速度
    
    // 观测噪声协方差矩阵
    double r_pos = 0.01;   // 位置观测噪声
    double r_rot = 0.01;   // 姿态观测噪声
    gtsam::Matrix R = gtsam::Matrix::Zero(6, 6);
    R(0, 0) = r_pos; R(1, 1) = r_pos; R(2, 2) = r_pos; // 位置
    R(3, 3) = r_rot; R(4, 4) = r_rot; R(5, 5) = r_rot; // 姿态
    
    // 误差协方差矩阵初始化
    gtsam::Matrix P = gtsam::Matrix::Identity(12, 12);
    
    // 卡尔曼滤波主循环
    for (size_t i = 0; i < poses.size(); ++i) {
        // 预测步骤
        if (i > 0) {
            x = F * x;  // 状态预测
            P = F * P * F.transpose() + Q;  // 误差协方差预测
        }
        
        // 更新步骤
        gtsam::Vector z(6); // 观测向量
        z[0] = poses[i].x(); // x
        z[1] = poses[i].y(); // y
        z[2] = poses[i].z(); // z
        gtsam::Rot3 rot = poses[i].rotation();
        gtsam::Vector3 rpy = rot.rpy();
        z[3] = rpy[0]; // roll
        z[4] = rpy[1]; // pitch
        z[5] = rpy[2]; // yaw
        
        gtsam::Vector y = z - H * x;  // 残差
        gtsam::Matrix S = H * P * H.transpose() + R;  // 残差协方差
        gtsam::Matrix K = P * H.transpose() * S.inverse();  // 卡尔曼增益
        
        x = x + K * y;  // 状态更新
        P = (gtsam::Matrix::Identity(12, 12) - K * H) * P;  // 误差协方差更新
        
        // 保存当前状态
        states.push_back(x);
    }
    
    return states;
}

Vector12 VelocityEstimator::fitLocalConstantVelocity(
    const std::vector<gtsam::Pose3>& poses,
    int startIndex,
    int endIndex) {
    
    int n = endIndex - startIndex + 1;
    if (n < 2) {
        return Vector12::Zero(12);
    }
    
    // 构建最小二乘问题
    // 模型: x(t) = x0 + vx * t, y(t) = y0 + vy * t, z(t) = z0 + vz * t
    //       roll(t) = roll0 + vroll * t, pitch(t) = pitch0 + vpitch * t, yaw(t) = yaw0 + vyaw * t
    // 其中 t 是相对于起始点的时间
    
    Eigen::MatrixXd A(n, 2);
    Eigen::VectorXd bx(n);
    Eigen::VectorXd by(n);
    Eigen::VectorXd bz(n);
    Eigen::VectorXd broll(n);
    Eigen::VectorXd bpitch(n);
    Eigen::VectorXd byaw(n);
    
    for (int i = 0; i < n; ++i) {
        double t = i * dt_;  // 时间
        A(i, 0) = 1.0;       // 常数项 (x0, y0, z0, roll0, pitch0, yaw0)
        A(i, 1) = t;         // 速度项 (vx, vy, vz, vroll, vpitch, vyaw)
        
        // 位置
        bx(i) = poses[startIndex + i].x();
        by(i) = poses[startIndex + i].y();
        bz(i) = poses[startIndex + i].z();
        
        // 姿态
        gtsam::Rot3 rot = poses[startIndex + i].rotation();
        gtsam::Vector3 rpy = rot.rpy();
        broll(i) = rpy[0];
        bpitch(i) = rpy[1];
        byaw(i) = rpy[2];
    }
    
    // 使用最小二乘法求解
    Eigen::Vector2d x = A.colPivHouseholderQr().solve(bx);
    Eigen::Vector2d y = A.colPivHouseholderQr().solve(by);
    Eigen::Vector2d z = A.colPivHouseholderQr().solve(bz);
    Eigen::Vector2d roll = A.colPivHouseholderQr().solve(broll);
    Eigen::Vector2d pitch = A.colPivHouseholderQr().solve(bpitch);
    Eigen::Vector2d yaw = A.colPivHouseholderQr().solve(byaw);
    
    // 返回速度分量 [vx, vy, vz, vroll, vpitch, vyaw]
    Vector12 velocity = Vector12::Zero(12);
    velocity[0] = x[1];      // vx
    velocity[1] = y[1];      // vy
    velocity[2] = z[1];      // vz
    velocity[3] = roll[1];   // vroll
    velocity[4] = pitch[1];  // vpitch
    velocity[5] = yaw[1];    // vyaw
    
    return velocity;
}

Vector12 VelocityEstimator::computeWeightedAverageVelocity(
    const std::vector<Vector12>& velocities,
    const std::vector<double>& weights) {
    
    if (velocities.empty() || weights.empty() || velocities.size() != weights.size()) {
        return Vector12::Zero(12);
    }
    
    // 归一化权重
    double weightSum = 0.0;
    for (double w : weights) {
        weightSum += w;
    }
    
    if (weightSum <= 0.0) {
        return Vector12::Zero(12);
    }
    
    // 计算加权平均
    Vector12 weightedVel = Vector12::Zero(12);
    for (size_t i = 0; i < velocities.size(); ++i) {
        double normalizedWeight = weights[i] / weightSum;
        weightedVel += normalizedWeight * velocities[i];
    }
    
    return weightedVel;
}