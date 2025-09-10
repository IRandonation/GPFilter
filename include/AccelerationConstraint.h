// include/AccelerationConstraint.h
#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>

class AccelerationConstraint : public gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4> {
public:
    using Key = gtsam::Key;
    using Vector4 = gtsam::Vector4;
    using Vector2 = gtsam::Vector2;
    using NoiseModel = gtsam::SharedNoiseModel;

    // 最大允许加速度（m/s²）
    double max_acceleration_;
    // 时间步长，用于计算加速度
    double dt_;

    /**
     * 构造函数
     * @param key1 第一个状态变量键（如S(i)）
     * @param key2 第二个状态变量键（如S(i+1)）
     * @param max_accel 最大加速度约束
     * @param dt 时间步长
     * @param model 噪声模型（通常为高斯模型）
     */
    AccelerationConstraint(Key key1, Key key2, double max_accel, double dt, const NoiseModel& model)
        : NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4>(model, key1, key2),
          max_acceleration_(max_accel), dt_(dt) {
    }

    /**
     * 计算误差项：加速度超过阈值时的惩罚项
     * @param state1 第一个状态（[x1, y1, vx1, vy1]）
     * @param state2 第二个状态（[x2, y2, vx2, vy2]）
     * @param H1 第一个状态的雅可比矩阵（可选）
     * @param H2 第二个状态的雅可比矩阵（可选）
     * @return 误差向量（加速度方向的缩放量）
     */
    gtsam::Vector evaluateError(const gtsam::Vector4& state1, const gtsam::Vector4& state2,
                               gtsam::OptionalMatrixType H1 = nullptr,
                               gtsam::OptionalMatrixType H2 = nullptr) const override {
        // 提取两个状态的速度分量
        Vector2 velocity1 = state1.tail(2);
        Vector2 velocity2 = state2.tail(2);
        
        // 计算实际的加速度：a = (v2 - v1) / dt
        Vector2 acceleration = (velocity2 - velocity1) / dt_;
        double accel_magnitude = acceleration.norm();
        
        // 若加速度超过阈值，误差为超出部分的向量；否则无误差
        Vector2 error = Vector2::Zero();
        if (accel_magnitude > max_acceleration_) {
            double excess = accel_magnitude - max_acceleration_;
            Vector2 accel_direction = acceleration.normalized();
            error = accel_direction * excess;
        }

        // 计算雅可比矩阵
        if (H1 || H2) {
            // 对state1的雅可比矩阵
            if (H1) {
                *H1 = gtsam::Matrix(2, 4);  // 2维误差，4维状态
                
                if (accel_magnitude > 1e-6) {
                    Vector2 accel_direction = acceleration.normalized();
                    double excess = (accel_magnitude > max_acceleration_) ? (accel_magnitude - max_acceleration_) : 0.0;
                    
                    // 误差对velocity1的导数
                    double derror_dv1x = -accel_direction[0] / dt_;
                    double derror_dv1y = -accel_direction[1] / dt_;
                    
                    if (excess > 0) {
                        // 考虑excess对导数的影响
                        double daccel_dv1x = -acceleration[0] / (accel_magnitude * dt_);
                        double daccel_dv1y = -acceleration[1] / (accel_magnitude * dt_);
                        
                        derror_dv1x += accel_direction[0] * daccel_dv1x * excess / accel_magnitude;
                        derror_dv1y += accel_direction[1] * daccel_dv1y * excess / accel_magnitude;
                    }
                    
                    *H1 << 0, 0, derror_dv1x, derror_dv1y,
                           0, 0, 0, derror_dv1y;  // 修正：第二行第一个元素应该是0
                } else {
                    *H1 << 0, 0, 0, 0,
                           0, 0, 0, 0;
                }
            }
            
            // 对state2的雅可比矩阵
            if (H2) {
                *H2 = gtsam::Matrix(2, 4);  // 2维误差，4维状态
                
                if (accel_magnitude > 1e-6) {
                    Vector2 accel_direction = acceleration.normalized();
                    double excess = (accel_magnitude > max_acceleration_) ? (accel_magnitude - max_acceleration_) : 0.0;
                    
                    // 误差对velocity2的导数
                    double derror_dv2x = accel_direction[0] / dt_;
                    double derror_dv2y = accel_direction[1] / dt_;
                    
                    if (excess > 0) {
                        // 考虑excess对导数的影响
                        double daccel_dv2x = acceleration[0] / (accel_magnitude * dt_);
                        double daccel_dv2y = acceleration[1] / (accel_magnitude * dt_);
                        
                        derror_dv2x += accel_direction[0] * daccel_dv2x * excess / accel_magnitude;
                        derror_dv2y += accel_direction[1] * daccel_dv2y * excess / accel_magnitude;
                    }
                    
                    *H2 << 0, 0, derror_dv2x, derror_dv2y,
                           0, 0, 0, derror_dv2y;  // 修正：第二行第一个元素应该是0
                } else {
                    *H2 << 0, 0, 0, 0,
                           0, 0, 0, 0;
                }
            }
        }
        // std::cout << "[DEBUG] 本因子误差向量: " << error << ", 范数: " << error.norm() << std::endl;

        return error;
    }
};