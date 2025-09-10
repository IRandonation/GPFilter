// include/AccelerationConstraint.h
#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>

// 前向声明12维状态向量类型
using Vector12 = gtsam::Vector;

class AccelerationConstraint : public gtsam::NoiseModelFactor2<Vector12, Vector12> {
public:
    using Key = gtsam::Key;
    using Vector3 = gtsam::Vector3;
    using NoiseModel = gtsam::SharedNoiseModel;

    // 最大允许线加速度（m/s²）
    double max_linear_acceleration_;
    // 最大允许角加速度（rad/s²）
    double max_angular_acceleration_;
    // 时间步长，用于计算加速度
    double dt_;

    /**
     * 构造函数
     * @param key1 第一个状态变量键（如S(i)）
     * @param key2 第二个状态变量键（如S(i+1)）
     * @param max_linear_accel 最大线加速度约束
     * @param max_angular_accel 最大角加速度约束
     * @param dt 时间步长
     * @param model 噪声模型（通常为高斯模型）
     */
    AccelerationConstraint(Key key1, Key key2, double max_linear_accel, double max_angular_accel, double dt, const NoiseModel& model)
        : NoiseModelFactor2<Vector12, Vector12>(model, key1, key2),
          max_linear_acceleration_(max_linear_accel),
          max_angular_acceleration_(max_angular_accel),
          dt_(dt) {
    }

    /**
     * 计算误差项：加速度超过阈值时的惩罚项
     * @param state1 第一个状态（[x1, y1, z1, roll1, pitch1, yaw1, vx1, vy1, vz1, vroll1, vpitch1, vyaw1]）
     * @param state2 第二个状态（[x2, y2, z2, roll2, pitch2, yaw2, vx2, vy2, vz2, vroll2, vpitch2, vyaw2]）
     * @param H1 第一个状态的雅可比矩阵（可选）
     * @param H2 第二个状态的雅可比矩阵（可选）
     * @return 误差向量（线加速度和角加速度方向的缩放量）
     */
    gtsam::Vector evaluateError(const Vector12& state1, const Vector12& state2,
                               gtsam::OptionalMatrixType H1 = nullptr,
                               gtsam::OptionalMatrixType H2 = nullptr) const override {
        // 提取两个状态的线速度分量
        Vector3 linear_velocity1 = state1.segment(6, 3);
        Vector3 linear_velocity2 = state2.segment(6, 3);
        
        // 提取两个状态的角速度分量
        Vector3 angular_velocity1 = state1.segment(9, 3);
        Vector3 angular_velocity2 = state2.segment(9, 3);
        
        // 计算实际的线加速度：a = (v2 - v1) / dt
        Vector3 linear_acceleration = (linear_velocity2 - linear_velocity1) / dt_;
        double linear_accel_magnitude = linear_acceleration.norm();
        
        // 计算实际的角加速度：alpha = (omega2 - omega1) / dt
        Vector3 angular_acceleration = (angular_velocity2 - angular_velocity1) / dt_;
        double angular_accel_magnitude = angular_acceleration.norm();
        
        // 误差向量：前3维为线加速度误差，后3维为角加速度误差
        gtsam::Vector error = gtsam::Vector::Zero(6);
        
        // 线加速度误差
        if (linear_accel_magnitude > max_linear_acceleration_) {
            double excess = linear_accel_magnitude - max_linear_acceleration_;
            Vector3 accel_direction = linear_acceleration.normalized();
            error.segment(0, 3) = accel_direction * excess;
        }
        
        // 角加速度误差
        if (angular_accel_magnitude > max_angular_acceleration_) {
            double excess = angular_accel_magnitude - max_angular_acceleration_;
            Vector3 accel_direction = angular_acceleration.normalized();
            error.segment(3, 3) = accel_direction * excess;
        }

        // 计算雅可比矩阵
        if (H1 || H2) {
            // 对state1的雅可比矩阵
            if (H1) {
                *H1 = gtsam::Matrix::Zero(6, 12);  // 6维误差，12维状态
                
                // 线加速度部分对state1的导数
                if (linear_accel_magnitude > 1e-6) {
                    Vector3 accel_direction = linear_acceleration.normalized();
                    double excess = (linear_accel_magnitude > max_linear_acceleration_) ?
                                   (linear_accel_magnitude - max_linear_acceleration_) : 0.0;
                    
                    // 误差对velocity1的导数
                    double derror_dv1x = -accel_direction[0] / dt_;
                    double derror_dv1y = -accel_direction[1] / dt_;
                    double derror_dv1z = -accel_direction[2] / dt_;
                    
                    if (excess > 0) {
                        // 考虑excess对导数的影响
                        double daccel_dv1x = -linear_acceleration[0] / (linear_accel_magnitude * dt_);
                        double daccel_dv1y = -linear_acceleration[1] / (linear_accel_magnitude * dt_);
                        double daccel_dv1z = -linear_acceleration[2] / (linear_accel_magnitude * dt_);
                        
                        derror_dv1x += accel_direction[0] * daccel_dv1x * excess / linear_accel_magnitude;
                        derror_dv1y += accel_direction[1] * daccel_dv1y * excess / linear_accel_magnitude;
                        derror_dv1z += accel_direction[2] * daccel_dv1z * excess / linear_accel_magnitude;
                    }
                    
                    (*H1)(0, 6) = derror_dv1x;  // vx1
                    (*H1)(1, 7) = derror_dv1y;  // vy1
                    (*H1)(2, 8) = derror_dv1z;  // vz1
                }
                
                // 角加速度部分对state1的导数
                if (angular_accel_magnitude > 1e-6) {
                    Vector3 accel_direction = angular_acceleration.normalized();
                    double excess = (angular_accel_magnitude > max_angular_acceleration_) ?
                                   (angular_accel_magnitude - max_angular_acceleration_) : 0.0;
                    
                    // 误差对angular_velocity1的导数
                    double derror_dv1x = -accel_direction[0] / dt_;
                    double derror_dv1y = -accel_direction[1] / dt_;
                    double derror_dv1z = -accel_direction[2] / dt_;
                    
                    if (excess > 0) {
                        // 考虑excess对导数的影响
                        double daccel_dv1x = -angular_acceleration[0] / (angular_accel_magnitude * dt_);
                        double daccel_dv1y = -angular_acceleration[1] / (angular_accel_magnitude * dt_);
                        double daccel_dv1z = -angular_acceleration[2] / (angular_accel_magnitude * dt_);
                        
                        derror_dv1x += accel_direction[0] * daccel_dv1x * excess / angular_accel_magnitude;
                        derror_dv1y += accel_direction[1] * daccel_dv1y * excess / angular_accel_magnitude;
                        derror_dv1z += accel_direction[2] * daccel_dv1z * excess / angular_accel_magnitude;
                    }
                    
                    (*H1)(3, 9) = derror_dv1x;  // vroll1
                    (*H1)(4, 10) = derror_dv1y; // vpitch1
                    (*H1)(5, 11) = derror_dv1z; // vyaw1
                }
            }
            
            // 对state2的雅可比矩阵
            if (H2) {
                *H2 = gtsam::Matrix::Zero(6, 12);  // 6维误差，12维状态
                
                // 线加速度部分对state2的导数
                if (linear_accel_magnitude > 1e-6) {
                    Vector3 accel_direction = linear_acceleration.normalized();
                    double excess = (linear_accel_magnitude > max_linear_acceleration_) ?
                                   (linear_accel_magnitude - max_linear_acceleration_) : 0.0;
                    
                    // 误差对velocity2的导数
                    double derror_dv2x = accel_direction[0] / dt_;
                    double derror_dv2y = accel_direction[1] / dt_;
                    double derror_dv2z = accel_direction[2] / dt_;
                    
                    if (excess > 0) {
                        // 考虑excess对导数的影响
                        double daccel_dv2x = linear_acceleration[0] / (linear_accel_magnitude * dt_);
                        double daccel_dv2y = linear_acceleration[1] / (linear_accel_magnitude * dt_);
                        double daccel_dv2z = linear_acceleration[2] / (linear_accel_magnitude * dt_);
                        
                        derror_dv2x += accel_direction[0] * daccel_dv2x * excess / linear_accel_magnitude;
                        derror_dv2y += accel_direction[1] * daccel_dv2y * excess / linear_accel_magnitude;
                        derror_dv2z += accel_direction[2] * daccel_dv2z * excess / linear_accel_magnitude;
                    }
                    
                    (*H2)(0, 6) = derror_dv2x;  // vx2
                    (*H2)(1, 7) = derror_dv2y;  // vy2
                    (*H2)(2, 8) = derror_dv2z;  // vz2
                }
                
                // 角加速度部分对state2的导数
                if (angular_accel_magnitude > 1e-6) {
                    Vector3 accel_direction = angular_acceleration.normalized();
                    double excess = (angular_accel_magnitude > max_angular_acceleration_) ?
                                   (angular_accel_magnitude - max_angular_acceleration_) : 0.0;
                    
                    // 误差对angular_velocity2的导数
                    double derror_dv2x = accel_direction[0] / dt_;
                    double derror_dv2y = accel_direction[1] / dt_;
                    double derror_dv2z = accel_direction[2] / dt_;
                    
                    if (excess > 0) {
                        // 考虑excess对导数的影响
                        double daccel_dv2x = angular_acceleration[0] / (angular_accel_magnitude * dt_);
                        double daccel_dv2y = angular_acceleration[1] / (angular_accel_magnitude * dt_);
                        double daccel_dv2z = angular_acceleration[2] / (angular_accel_magnitude * dt_);
                        
                        derror_dv2x += accel_direction[0] * daccel_dv2x * excess / angular_accel_magnitude;
                        derror_dv2y += accel_direction[1] * daccel_dv2y * excess / angular_accel_magnitude;
                        derror_dv2z += accel_direction[2] * daccel_dv2z * excess / angular_accel_magnitude;
                    }
                    
                    (*H2)(3, 9) = derror_dv2x;  // vroll2
                    (*H2)(4, 10) = derror_dv2y; // vpitch2
                    (*H2)(5, 11) = derror_dv2z; // vyaw2
                }
            }
        }

        return error;
    }
};