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

std::vector<gtsam::Vector4> VelocityEstimator::estimateVelocities(
    const std::vector<gtsam::Point2>& positions) {
    
    switch (method_) {
        case SIMPLE_DIFFERENCE:
            return estimateWithSimpleDifference(positions);
        case CONSTANT_VELOCITY:
            return estimateWithConstantVelocity(positions);
        case KALMAN_FILTER:
            return estimateWithKalmanFilter(positions);
        default:
            return estimateWithConstantVelocity(positions); // 默认使用匀速模型
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

std::vector<gtsam::Vector4> VelocityEstimator::estimateWithSimpleDifference(
    const std::vector<gtsam::Point2>& positions) {
    
    std::vector<gtsam::Vector4> states;
    
    for (size_t i = 0; i < positions.size(); ++i) {
        gtsam::Vector4 state;
        state << positions[i].x(), positions[i].y(), 0.0, 0.0;
        
        // 对非第一个点，用前一位置的差分估算速度
        if (i > 0) {
            double dx = positions[i].x() - positions[i-1].x();
            double dy = positions[i].y() - positions[i-1].y();
            state[2] = dx / dt_; // vx = Δx / Δt
            state[3] = dy / dt_; // vy = Δy / Δt
        }
        
        states.push_back(state);
    }
    
    return states;
}

std::vector<gtsam::Vector4> VelocityEstimator::estimateWithConstantVelocity(
    const std::vector<gtsam::Point2>& positions) {
    
    std::vector<gtsam::Vector4> states;
    int halfWindow = windowSize_ / 2;
    
    for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
        gtsam::Vector4 state;
        state << positions[i].x(), positions[i].y(), 0.0, 0.0;
        
        // 确定局部窗口的起始和结束索引
        int startIdx = std::max(0, i - halfWindow);
        int endIdx = std::min(static_cast<int>(positions.size()) - 1, i + halfWindow);
        
        // 确保窗口内有足够的点
        if (endIdx - startIdx >= 2) {
            // 使用最小二乘法拟合局部匀速模型
            gtsam::Vector2 velocity = fitLocalConstantVelocity(positions, startIdx, endIdx);
            state[2] = velocity[0]; // vx
            state[3] = velocity[1]; // vy
        } else if (i > 0) {
            // 如果窗口内点数不足，回退到简单差分法
            double dx = positions[i].x() - positions[i-1].x();
            double dy = positions[i].y() - positions[i-1].y();
            state[2] = dx / dt_; // vx = Δx / Δt
            state[3] = dy / dt_; // vy = Δy / Δt
        }
        
        states.push_back(state);
    }
    
    return states;
}

std::vector<gtsam::Vector4> VelocityEstimator::estimateWithKalmanFilter(
    const std::vector<gtsam::Point2>& positions) {
    
    // 状态向量: [x, y, vx, vy]
    // 观测向量: [x, y]
    
    std::vector<gtsam::Vector4> states;
    
    // 初始化卡尔曼滤波参数
    gtsam::Vector4 x; // 状态向量
    x << positions[0].x(), positions[0].y(), 0.0, 0.0; // 初始状态
    
    // 状态转移矩阵 (匀速模型)
    gtsam::Matrix4 F;
    F << 1, 0, dt_, 0,
         0, 1, 0, dt_,
         0, 0, 1, 0,
         0, 0, 0, 1;
    
    // 观测矩阵
    gtsam::Matrix H = gtsam::Matrix::Zero(2, 4);
    H(0, 0) = 1; // 观测x
    H(1, 1) = 1; // 观测y
    
    // 过程噪声协方差矩阵
    double q_pos = 0.01;   // 位置过程噪声
    double q_vel = 0.1;    // 速度过程噪声
    gtsam::Matrix4 Q;
    Q << q_pos, 0, 0, 0,
         0, q_pos, 0, 0,
         0, 0, q_vel, 0,
         0, 0, 0, q_vel;
    
    // 观测噪声协方差矩阵
    double r_pos = 0.01;   // 位置观测噪声
    gtsam::Matrix2 R = r_pos * gtsam::Matrix2::Identity();
    
    // 误差协方差矩阵初始化
    gtsam::Matrix4 P = gtsam::Matrix4::Identity();
    
    // 卡尔曼滤波主循环
    for (size_t i = 0; i < positions.size(); ++i) {
        // 预测步骤
        if (i > 0) {
            x = F * x;  // 状态预测
            P = F * P * F.transpose() + Q;  // 误差协方差预测
        }
        
        // 更新步骤
        gtsam::Vector2 z; // 观测向量
        z << positions[i].x(), positions[i].y();
        
        gtsam::Vector2 y = z - H * x;  // 残差
        gtsam::Matrix2 S = H * P * H.transpose() + R;  // 残差协方差
        gtsam::Matrix K = P * H.transpose() * S.inverse();  // 卡尔曼增益
        
        x = x + K * y;  // 状态更新
        P = (gtsam::Matrix4::Identity() - K * H) * P;  // 误差协方差更新
        
        // 保存当前状态
        states.push_back(x);
    }
    
    return states;
}

gtsam::Vector2 VelocityEstimator::fitLocalConstantVelocity(
    const std::vector<gtsam::Point2>& positions,
    int startIndex, 
    int endIndex) {
    
    int n = endIndex - startIndex + 1;
    if (n < 2) {
        return gtsam::Vector2::Zero();
    }
    
    // 构建最小二乘问题
    // 模型: x(t) = x0 + vx * t, y(t) = y0 + vy * t
    // 其中 t 是相对于起始点的时间
    
    Eigen::MatrixXd A(n, 2);
    Eigen::VectorXd bx(n);
    Eigen::VectorXd by(n);
    
    for (int i = 0; i < n; ++i) {
        double t = i * dt_;  // 时间
        A(i, 0) = 1.0;       // 常数项 (x0, y0)
        A(i, 1) = t;         // 速度项 (vx, vy)
        
        bx(i) = positions[startIndex + i].x();
        by(i) = positions[startIndex + i].y();
    }
    
    // 使用最小二乘法求解
    Eigen::Vector2d x = A.colPivHouseholderQr().solve(bx);
    Eigen::Vector2d y = A.colPivHouseholderQr().solve(by);
    
    // 返回速度分量
    return gtsam::Vector2(x[1], y[1]);
}

gtsam::Vector2 VelocityEstimator::computeWeightedAverageVelocity(
    const std::vector<gtsam::Vector2>& velocities,
    const std::vector<double>& weights) {
    
    if (velocities.empty() || weights.empty() || velocities.size() != weights.size()) {
        return gtsam::Vector2::Zero();
    }
    
    // 归一化权重
    double weightSum = 0.0;
    for (double w : weights) {
        weightSum += w;
    }
    
    if (weightSum <= 0.0) {
        return gtsam::Vector2::Zero();
    }
    
    // 计算加权平均
    gtsam::Vector2 weightedVel = gtsam::Vector2::Zero();
    for (size_t i = 0; i < velocities.size(); ++i) {
        double normalizedWeight = weights[i] / weightSum;
        weightedVel += normalizedWeight * velocities[i];
    }
    
    return weightedVel;
}