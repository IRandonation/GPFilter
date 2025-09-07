// include/AccelerationConstraint.h
#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>

class AccelerationConstraint : public gtsam::NoiseModelFactor1<gtsam::Vector4> {
public:
    using Key = gtsam::Key;
    using Vector4 = gtsam::Vector4;
    using NoiseModel = gtsam::SharedNoiseModel;

    // 最大允许加速度（m/s²）
    double max_acceleration_;

    /**
     * 构造函数
     * @param key 状态变量键（如S(0)）
     * @param max_accel 最大加速度约束
     * @param model 噪声模型（通常为高斯模型）
     */
    AccelerationConstraint(Key key, double max_accel, const NoiseModel& model)
        : NoiseModelFactor1<gtsam::Vector4>(model, key), max_acceleration_(max_accel) {
    }

    /**
     * 计算误差项：加速度超过阈值时的惩罚项
     * @param state 当前状态（[x, y, vx, vy]）
     * @param H 雅可比矩阵（可选，用于优化效率）
     * @return 误差向量（加速度方向的缩放量）
     */
    gtsam::Vector evaluateError(const gtsam::Vector4& state,
                               gtsam::OptionalMatrixType H = nullptr) const override {
        // 提取速度分量（vx, vy）
        gtsam::Vector2 velocity = state.tail(2);
        double speed = velocity.norm();
        
        // 假设加速度方向与速度方向相同，计算加速度
        // 这里简化处理，实际应用中可能需要更复杂的加速度计算
        gtsam::Vector2 acceleration = velocity * 0.1; // 简化的加速度计算
        double accel_magnitude = acceleration.norm();

        // 若加速度超过阈值，误差为加速度方向的缩放量；否则无误差
        gtsam::Vector error = gtsam::Vector2::Zero();
        if (accel_magnitude > max_acceleration_) {
            error = acceleration * (accel_magnitude - max_acceleration_) / accel_magnitude;
        }

        // 计算雅可比矩阵（状态对误差的导数）
        if (H) {
            *H = gtsam::Matrix(2, 4);  // 2维误差，4维状态
            *H << 0, 0, 0.1, 0,      // vx对误差的导数（简化处理）
                  0, 0, 0, 0.1;      // vy对误差的导数（简化处理）
        }

        return error;
    }
};