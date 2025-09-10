// GPFactor.h
#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>

class GPFactor : public gtsam::NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4> {
public:
    GPFactor(gtsam::Key key1, gtsam::Key key2, double dt,
             const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor2<gtsam::Vector4, gtsam::Vector4>(model, key1, key2), dt_(dt) {
        F_ = (gtsam::Matrix4() <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1).finished();
    }

    gtsam::Vector evaluateError(
        const gtsam::Vector4& x1, const gtsam::Vector4& x2,
        gtsam::OptionalMatrixType H1 = nullptr, gtsam::OptionalMatrixType H2 = nullptr) const override {
        
        if (H1) {
            // 雅可比矩阵应该是误差向量(4维)对状态变量(4维)的导数
            // H1 是误差对 x1 的导数，维度为 4x4
            // 误差函数: error = x2 - F_ * x1
            // 对 x1 的导数是 -F_
            *H1 = -F_;
        }
        
        if (H2) {
            // H2 是误差对 x2 的导数，维度为 4x4
            // 误差函数: error = x2 - F_ * x1
            // 对 x2 的导数是单位矩阵
            *H2 = gtsam::Matrix4::Identity();
        }
        // gtsam::Vector error = x2 - F_ * x1;

        // std::cout << "[DEBUG] 本因子误差向量: " << error << ", 范数: " << error.norm() << std::endl;
        
        return x2 - F_ * x1;
    }

private:
    double dt_;
    gtsam::Matrix4 F_;
};