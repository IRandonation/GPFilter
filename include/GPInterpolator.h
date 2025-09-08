#pragma once
#include <gtsam/base/Vector.h>
#include <vector>
#include <gtsam/base/Matrix.h>

class GPInterpolator {
private:
    double dt_;                   // 原始数据时间间隔
    int interpolationFactor_;     // 插值因子
    double interpolatedDt_;       // 插值后时间间隔
    double lengthScale_;          // 核函数特征长度（控制平滑度）
    double noiseSigma_;           // 观测噪声标准差（控制与原始点的接近程度）

    // 核函数：平方指数核，定义时间点之间的相关性
    double kernel(double t1, double t2) const;

    // 构建协方差矩阵
    gtsam::Matrix buildCovarianceMatrix(const std::vector<double>& times) const;

    // GP后验估计
    gtsam::Vector4 gpPosteriorEstimate(
        double t, 
        const std::vector<double>& originalTimes,
        const std::vector<gtsam::Vector4>& originalStates,
        const gtsam::Matrix& KXX_inv) const;

public:
    // 构造函数（保持原有接口）
    GPInterpolator(double dt, int interpolationFactor);

    // 执行GP插值（保持原有接口）
    std::vector<gtsam::Vector4> interpolate(
        const std::vector<gtsam::Vector4>& originalTrajectory);

    // 设置插值因子
    void setInterpolationFactor(int factor);

    // 获取插值因子
    int getInterpolationFactor() const;

    // 获取插值后时间间隔
    double getInterpolatedDt() const;

    // 设置核函数特征长度（越大越平滑）
    void setLengthScale(double l);

    // 设置观测噪声（越小越接近原始数据）
    void setNoiseSigma(double sigma);
};
    