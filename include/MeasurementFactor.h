// include/MeasurementFactor.h
#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

// 前向声明12维状态向量类型
using Vector12 = gtsam::Vector;

class MeasurementFactor : public gtsam::NoiseModelFactor1<Vector12> {
public:
    gtsam::Pose3 measured_;
    
    MeasurementFactor(gtsam::Key key, const gtsam::Pose3& measured,
                    const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor1<Vector12>(model, key), measured_(measured) {
    }
    
    gtsam::Vector evaluateError(const Vector12& state,
                               gtsam::OptionalMatrixType H = nullptr) const override {
        if (H) {
            // 雅可比矩阵：误差向量(6维)对状态变量(12维)的导数
            // 只对位置和姿态部分有导数，速度部分导数为0
            *H = gtsam::Matrix::Zero(6, 12);
            // 位置部分导数
            (*H)(0, 0) = 1;  // x
            (*H)(1, 1) = 1;  // y
            (*H)(2, 2) = 1;  // z
            // 姿态部分导数
            (*H)(3, 3) = 1;  // roll
            (*H)(4, 4) = 1;  // pitch
            (*H)(5, 5) = 1;  // yaw
        }
        
        // 比较位置和姿态部分，不比较速度部分
        gtsam::Vector error(6);
        
        // 位置误差
        error[0] = state[0] - measured_.x();     // x
        error[1] = state[1] - measured_.y();     // y
        error[2] = state[2] - measured_.z();     // z
        
        // 姿态误差
        gtsam::Rot3 measuredRot = measured_.rotation();
        gtsam::Vector3 measuredRPY = measuredRot.rpy();
        error[3] = state[3] - measuredRPY[0];   // roll
        error[4] = state[4] - measuredRPY[1];   // pitch
        error[5] = state[5] - measuredRPY[2];   // yaw
        
        return error;
    }
};