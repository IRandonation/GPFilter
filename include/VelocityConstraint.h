// include/VelocityConstraint.h
#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>

// 前向声明12维状态向量类型
using Vector12 = gtsam::Vector;

class VelocityConstraint : public gtsam::NoiseModelFactor1<Vector12> {
public:
    using Key = gtsam::Key;
    using NoiseModel = gtsam::SharedNoiseModel;

    // 最大允许线速度（m/s）
    double max_linear_velocity_;
    // 最大允许角速度（rad/s）
    double max_angular_velocity_;

    /**
     * 构造函数
     * @param key 状态变量键（如S(0)）
     * @param max_linear_vel 最大线速度约束
     * @param max_angular_vel 最大角速度约束
     * @param model 噪声模型（通常为高斯模型）
     */
    VelocityConstraint(Key key, double max_linear_vel, double max_angular_vel, const NoiseModel& model)
        : NoiseModelFactor1<Vector12>(model, key),
          max_linear_velocity_(max_linear_vel),
          max_angular_velocity_(max_angular_vel) {
    }

    /**
     * 计算误差项：速度超过阈值时的惩罚项
     * @param state 当前状态（[x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]）
     * @param H 雅可比矩阵（可选，用于优化效率）
     * @return 误差向量（线速度和角速度方向的缩放量）
     */
    gtsam::Vector evaluateError(const Vector12& state,
                               gtsam::OptionalMatrixType H = nullptr) const override {
        // 提取线速度分量（vx, vy, vz）
        gtsam::Vector3 linear_velocity = state.segment(6, 3);
        double linear_speed = linear_velocity.norm();
        
        // 提取角速度分量（vroll, vpitch, vyaw）
        gtsam::Vector3 angular_velocity = state.segment(9, 3);
        double angular_speed = angular_velocity.norm();

        // 误差向量：前3维为线速度误差，后3维为角速度误差
        gtsam::Vector error = gtsam::Vector::Zero(6);
        
        // 线速度误差
        if (linear_speed > max_linear_velocity_) {
            error.segment(0, 3) = linear_velocity * (linear_speed - max_linear_velocity_) / linear_speed;
        }
        
        // 角速度误差
        if (angular_speed > max_angular_velocity_) {
            error.segment(3, 3) = angular_velocity * (angular_speed - max_angular_velocity_) / angular_speed;
        }

        // 计算雅可比矩阵（状态对误差的导数）
        if (H) {
            *H = gtsam::Matrix::Zero(6, 12);  // 6维误差，12维状态
            
            // 线速度部分对误差的导数
            if (linear_speed > max_linear_velocity_) {
                (*H)(0, 6) = 1;  // vx对误差的导数
                (*H)(1, 7) = 1;  // vy对误差的导数
                (*H)(2, 8) = 1;  // vz对误差的导数
            }
            
            // 角速度部分对误差的导数
            if (angular_speed > max_angular_velocity_) {
                (*H)(3, 9) = 1;  // vroll对误差的导数
                (*H)(4, 10) = 1; // vpitch对误差的导数
                (*H)(5, 11) = 1; // vyaw对误差的导数
            }
        }
        
        return error;
    }
};