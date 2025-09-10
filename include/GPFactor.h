// GPFactor.h
#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>

// 前向声明12维状态向量类型
using Vector12 = gtsam::Vector;

class GPFactor : public gtsam::NoiseModelFactor2<Vector12, Vector12> {
public:
    GPFactor(gtsam::Key key1, gtsam::Key key2, double dt,
             const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor2<Vector12, Vector12>(model, key1, key2), dt_(dt) {
        // 构建12维状态转移矩阵
        // 状态向量: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
        F_ = gtsam::Matrix::Zero(12, 12);
        
        // 位置部分: x = x + vx * dt
        F_(0, 0) = 1; F_(0, 6) = dt;
        // 位置部分: y = y + vy * dt
        F_(1, 1) = 1; F_(1, 7) = dt;
        // 位置部分: z = z + vz * dt
        F_(2, 2) = 1; F_(2, 8) = dt;
        
        // 姿态部分: roll = roll + vroll * dt
        F_(3, 3) = 1; F_(3, 9) = dt;
        // 姿态部分: pitch = pitch + vpitch * dt
        F_(4, 4) = 1; F_(4, 10) = dt;
        // 姿态部分: yaw = yaw + vyaw * dt
        F_(5, 5) = 1; F_(5, 11) = dt;
        
        // 速度部分保持不变
        F_(6, 6) = 1;  // vx
        F_(7, 7) = 1;  // vy
        F_(8, 8) = 1;  // vz
        F_(9, 9) = 1;  // vroll
        F_(10, 10) = 1; // vpitch
        F_(11, 11) = 1; // vyaw
    }

    gtsam::Vector evaluateError(
        const Vector12& x1, const Vector12& x2,
        gtsam::OptionalMatrixType H1 = nullptr, gtsam::OptionalMatrixType H2 = nullptr) const override {
        
        if (H1) {
            // 雅可比矩阵应该是误差向量(12维)对状态变量(12维)的导数
            // H1 是误差对 x1 的导数，维度为 12x12
            // 误差函数: error = x2 - F_ * x1
            // 对 x1 的导数是 -F_
            *H1 = -F_;
        }
        
        if (H2) {
            // H2 是误差对 x2 的导数，维度为 12x12
            // 误差函数: error = x2 - F_ * x1
            // 对 x2 的导数是单位矩阵
            *H2 = gtsam::Matrix::Identity(12, 12);
        }
        
        // 计算预测状态和实际状态之间的误差
        return x2 - F_ * x1;
    }

private:
    double dt_;
    gtsam::Matrix F_;  // 12x12 状态转移矩阵
};