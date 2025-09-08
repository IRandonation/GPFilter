// GPInterpolator.cpp
#include "GPInterpolator.h"
#include <gtsam/base/Matrix.h>
#include <iostream>

GPInterpolator::GPInterpolator(double dt, int interpolationFactor)
    : dt_(dt), interpolationFactor_(interpolationFactor) {
    interpolatedDt_ = dt_ / interpolationFactor_;
}

std::vector<gtsam::Vector4> GPInterpolator::interpolate(
    const std::vector<gtsam::Vector4>& originalTrajectory) {
    
    std::vector<gtsam::Vector4> interpolatedTrajectory;
    
    if (originalTrajectory.empty()) {
        return interpolatedTrajectory;
    }
    
    // 添加第一个点
    interpolatedTrajectory.push_back(originalTrajectory[0]);
    
    // 对每对相邻点进行插值
    for (size_t i = 0; i < originalTrajectory.size() - 1; ++i) {
        const gtsam::Vector4& state1 = originalTrajectory[i];
        const gtsam::Vector4& state2 = originalTrajectory[i + 1];
        
        // 在两个原始点之间插值
        for (int j = 1; j < interpolationFactor_; ++j) {
            double alpha = static_cast<double>(j) / interpolationFactor_;
            gtsam::Vector4 interpolatedPoint = gpRegressionInterpolation(state1, state2, alpha);
            interpolatedTrajectory.push_back(interpolatedPoint);
        }
        
        // 添加下一个原始点
        interpolatedTrajectory.push_back(state2);
    }
    
    return interpolatedTrajectory;
}


void GPInterpolator::setInterpolationFactor(int factor) {
    interpolationFactor_ = factor;
    interpolatedDt_ = dt_ / interpolationFactor_;
}

int GPInterpolator::getInterpolationFactor() const {
    return interpolationFactor_;
}

double GPInterpolator::getInterpolatedDt() const {
    return interpolatedDt_;
}

gtsam::Matrix4 GPInterpolator::computeStateTransitionMatrix(double dt) const {
    // 状态转移矩阵 F = [I dt*I; 0 I]
    gtsam::Matrix4 F;
    F << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;
    return F;
}

gtsam::Matrix4 GPInterpolator::computeProcessNoiseCovariance(double dt) const {
    // 过程噪声协方差矩阵 Q
    // 简化模型，假设加速度噪声
    double q = 0.01; // 降低过程噪声强度，使插值结果更平滑
    gtsam::Matrix4 Q;
    double dt2 = dt * dt;
    double dt3 = dt2 * dt / 2.0;
    double dt4 = dt3 * dt / 3.0;
    
    Q << dt4 * q, 0, dt3 * q, 0,
         0, dt4 * q, 0, dt3 * q,
         dt3 * q, 0, dt2 * q, 0,
         0, dt3 * q, 0, dt2 * q;
    
    return Q;
}

gtsam::Vector4 GPInterpolator::gpRegressionInterpolation(
    const gtsam::Vector4& state1, const gtsam::Vector4& state2, double alpha) {
    
    double dt = dt_;
    double t1 = 0.0;
    double t2 = dt;
    double t = t1 + alpha * (t2 - t1);
    
    // 计算状态转移矩阵
    gtsam::Matrix4 F_t1 = computeStateTransitionMatrix(t1);
    gtsam::Matrix4 F_t2 = computeStateTransitionMatrix(t2);
    gtsam::Matrix4 F_t = computeStateTransitionMatrix(t);
    
    // 计算过程噪声协方差
    gtsam::Matrix4 Q_t1 = computeProcessNoiseCovariance(t1);
    gtsam::Matrix4 Q_t2 = computeProcessNoiseCovariance(t2);
    gtsam::Matrix4 Q_t = computeProcessNoiseCovariance(t);
    
    // 计算协方差矩阵
    gtsam::Matrix4 Sigma11 = Q_t1;
    gtsam::Matrix4 Sigma22 = Q_t2;
    gtsam::Matrix4 Sigma12 = F_t1 * Q_t; // 简化计算
    
    // 高斯过程回归插值
    // x_t = F_t1_to_t * x1 + K * (x2 - F_t1_to_t2 * x1)
    // 其中 K = Sigma_t1_to_t * Sigma_t1_to_t2^-1
    
    gtsam::Matrix4 F_t1_to_t = computeStateTransitionMatrix(t - t1);
    gtsam::Matrix4 F_t1_to_t2 = computeStateTransitionMatrix(t2 - t1);
    
    // 计算卡尔曼增益
    gtsam::Matrix4 K = Sigma12 * Sigma22.inverse();
    
    // 插值状态
    gtsam::Vector4 predicted_state = F_t1_to_t * state1;
    gtsam::Vector4 innovation = state2 - F_t1_to_t2 * state1;
    gtsam::Vector4 interpolated_state = predicted_state + K * innovation;
    
    // 应用更严格的加速度约束
    double max_acceleration = 2.0; // 最大加速度限制
    double interpolated_dt = dt / interpolationFactor_;
    
    // 计算插值点的速度
    gtsam::Vector2 interpolated_velocity = interpolated_state.tail(2);
    double interpolated_speed = interpolated_velocity.norm();
    
    // 计算加速度（基于速度变化）
    gtsam::Vector2 velocity1 = state1.tail(2);
    gtsam::Vector2 velocity2 = state2.tail(2);
    
    // 估计加速度：基于速度变化和时间
    gtsam::Vector2 accel_estimate = (velocity2 - velocity1) / dt;
    double accel_magnitude = accel_estimate.norm();
    
    // 如果估计的加速度超过限制，则使用更保守的插值方法
    if (accel_magnitude > max_acceleration) {
        // 使用更平滑的插值方法：限制速度变化率
        double max_velocity_change = max_acceleration * interpolated_dt;
        
        // 计算从state1到当前插值点的允许速度变化
        gtsam::Vector2 direction_from_state1 = (interpolated_velocity - velocity1);
        double change_from_state1 = direction_from_state1.norm();
        
        if (change_from_state1 > max_velocity_change) {
            // 限制速度变化
            double scale_factor = max_velocity_change / change_from_state1;
            gtsam::Vector2 limited_velocity = velocity1 + direction_from_state1 * scale_factor;
            
            // 更新插值状态
            interpolated_state[2] = limited_velocity[0];
            interpolated_state[3] = limited_velocity[1];
        }
        
        // 同时确保到state2的速度变化也在限制内
        gtsam::Vector2 direction_to_state2 = (velocity2 - interpolated_state.tail(2));
        double change_to_state2 = direction_to_state2.norm();
        
        if (change_to_state2 > max_velocity_change) {
            // 限制速度变化
            double scale_factor = max_velocity_change / change_to_state2;
            gtsam::Vector2 limited_velocity = interpolated_state.tail(2) + direction_to_state2 * scale_factor;
            
            // 更新插值状态
            interpolated_state[2] = limited_velocity[0];
            interpolated_state[3] = limited_velocity[1];
        }
    }
    
    // 额外的安全检查：确保插值速度不会过大
    double max_safe_speed = 5.0; // 最大安全速度
    gtsam::Vector2 final_velocity = interpolated_state.tail(2);
    double final_speed = final_velocity.norm();
    
    if (final_speed > max_safe_speed) {
        // 限制速度
        double scale_factor = max_safe_speed / final_speed;
        interpolated_state[2] = final_velocity[0] * scale_factor;
        interpolated_state[3] = final_velocity[1] * scale_factor;
    }
    
    return interpolated_state;
}