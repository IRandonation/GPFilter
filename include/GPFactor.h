// GPFactor.h
#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>

// 前向声明6维状态向量类型
using Vector6 = gtsam::Vector;

class GPFactor : public gtsam::NoiseModelFactor2<Vector6, Vector6> {
public:
    GPFactor(gtsam::Key key1, gtsam::Key key2, double dt,
             const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor2<Vector6, Vector6>(model, key1, key2), dt_(dt) {
        // 构建6维状态转移矩阵
        // 状态向量: [x, y, z, roll, pitch, yaw]
        // 使用简单的恒等模型，假设相邻状态之间的变化很小
        F_ = gtsam::Matrix::Identity(6, 6);
    }

    gtsam::Vector evaluateError(
        const Vector6& x1, const Vector6& x2,
        gtsam::OptionalMatrixType H1 = nullptr, gtsam::OptionalMatrixType H2 = nullptr) const override {
        
        if (H1) {
            // 雅可比矩阵应该是误差向量(6维)对状态变量(6维)的导数
            // H1 是误差对 x1 的导数，维度为 6x6
            // 误差函数: error = x2 - F_ * x1
            // 对 x1 的导数是 -F_
            *H1 = -F_;
        }
        
        if (H2) {
            // H2 是误差对 x2 的导数，维度为 6x6
            // 误差函数: error = x2 - F_ * x1
            // 对 x2 的导数是单位矩阵
            *H2 = gtsam::Matrix::Identity(6, 6);
        }
        
        // 计算预测状态和实际状态之间的误差
        return x2 - F_ * x1;
    }

private:
    double dt_;
    gtsam::Matrix F_;  // 6x6 状态转移矩阵
};