// include/VelocityConstraint.h
#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Vector.h>

class VelocityConstraint : public gtsam::NoiseModelFactor1<gtsam::Vector4> {
public:
    using Key = gtsam::Key;
    using Vector4 = gtsam::Vector4;
    using NoiseModel = gtsam::SharedNoiseModel;

    // 最大允许速度（m/s）
    double max_velocity_;

    /**
     * 构造函数
     * @param key 状态变量键（如S(0)）
     * @param max_vel 最大速度约束
     * @param model 噪声模型（通常为高斯模型）
     */
    VelocityConstraint(Key key, double max_vel, const NoiseModel& model)
        : NoiseModelFactor1<gtsam::Vector4>(model, key), max_velocity_(max_vel) {
    }

    /**
     * 计算误差项：速度超过阈值时的惩罚项
     * @param state 当前状态（[x, y, vx, vy]）
     * @param H 雅可比矩阵（可选，用于优化效率）
     * @return 误差向量（速度方向的缩放量）
     */
    gtsam::Vector evaluateError(const gtsam::Vector4& state,
                               gtsam::OptionalMatrixType H = nullptr) const override {
        // 提取速度分量（vx, vy）
        gtsam::Vector2 velocity = state.tail(2);
        double speed = velocity.norm();

        // 若速度超过阈值，误差为速度方向的缩放量；否则无误差
        gtsam::Vector error = gtsam::Vector2::Zero();
        if (speed > max_velocity_) {
            error = velocity * (speed - max_velocity_) / speed;
        }

        // 计算雅可比矩阵（状态对误差的导数）
        if (H) {
            *H = gtsam::Matrix(2, 4);  // 2维误差，4维状态
            *H << 0, 0, 1, 0,         // vx对误差的导数
                  0, 0, 0, 1;         // vy对误差的导数
        }
        // std::cout << "[DEBUG] 本因子误差向量: " << error << ", 范数: " << error.norm() << std::endl;
        return error;
    }
};