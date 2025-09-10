#include "GPInterpolator.h"
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

// 构造函数：保持原有接口，初始化参数
GPInterpolator::GPInterpolator(double dt, int interpolationFactor)
    : dt_(dt), interpolationFactor_(interpolationFactor) {
    interpolatedDt_ = dt_ / interpolationFactor_;
    lengthScale_ = dt_;  // 默认特征长度为原始时间间隔，平衡平滑性与跟踪性
    noiseSigma_ = 0.01;  // 默认观测噪声较小，保证与原始轨迹接近
    inv_lengthScale_sq_ = 1.0 / (lengthScale_ * lengthScale_);
}


// 核函数：平方指数核，控制时间点之间的相关性
double GPInterpolator::kernel(double t1, double t2) const {
    double dt = t1 - t2;
    return exp(-0.5 * (dt * dt) * inv_lengthScale_sq_);
}

// 构建原始点之间的协方差矩阵
gtsam::Matrix GPInterpolator::buildCovarianceMatrix(const std::vector<double>& times) const {
    int n = times.size();
    gtsam::Matrix K(n, n);
    
    for (int i = 0; i < n; ++i) {
        // 只计算上三角部分（i <= j），再复制到下三角
        for (int j = i; j < n; ++j) {  // j从i开始，避免重复计算
            double dt = times[i] - times[j];
            K(i, j) = kernel(times[i], times[j]);
            if (i == j) {
                K(i, j) += noiseSigma_ * noiseSigma_;  // 对角线加噪声
            } else {
                K(j, i) = K(i, j);  // 对称复制
            }
        }
    }
    return K;
}

// GP后验估计：计算任意时间点t的状态估计
Vector12 GPInterpolator::gpPosteriorEstimate(
    double t,
    const std::vector<double>& originalTimes,
    const std::vector<Vector12>& originalStates,
    const gtsam::Matrix& KXX_inv) const {
    
    int n = originalTimes.size();
    
    // 计算当前时间t与所有原始时间点的协方差向量
    gtsam::Vector k(n);
    for (int i = 0; i < n; ++i) {
        k(i) = kernel(t, originalTimes[i]);
    }
    
    // 计算权重向量：K(x,X) * K(X,X)^{-1}
    gtsam::Vector weights = KXX_inv * k;
    
    // 对状态的12个维度（x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw）分别进行估计
    Vector12 result = Vector12::Zero(12);
    for (int dim = 0; dim < 12; ++dim) {
        // 构建该维度的观测向量
        gtsam::Vector y(n);
        for (int i = 0; i < n; ++i) {
            y(i) = originalStates[i](dim);
        }
        // 后验均值 = 权重 * 观测向量
        result(dim) = weights.dot(y);
    }
    
    return result;
}

// 插值主函数：保持原有输入输出格式
std::vector<Vector12> GPInterpolator::interpolate(
    const std::vector<Vector12>& originalTrajectory) {
    
    std::vector<Vector12> interpolatedTrajectory;
    
    if (originalTrajectory.empty()) {
        return interpolatedTrajectory;
    }
    
    // 为原始轨迹分配时间戳（0, dt, 2*dt, ...）
    std::vector<double> originalTimes;
    for (size_t i = 0; i < originalTrajectory.size(); ++i) {
        originalTimes.push_back(i * dt_);
    }
    
    // 生成插值时间点序列
    std::vector<double> interpolatedTimes;
    double totalTime = (originalTrajectory.size() - 1) * dt_;
    for (double t = 0; t <= totalTime; t += interpolatedDt_) {
        interpolatedTimes.push_back(t);
    }
    
    // 构建并求逆协方差矩阵（只计算一次，提高效率）
    int n = originalTimes.size();
    gtsam::Matrix KXX = buildCovarianceMatrix(originalTimes);
    // 对正定矩阵进行Cholesky分解（LDLT更稳定，支持数值扰动）
    Eigen::LDLT<gtsam::Matrix> ldlt(KXX);
    // 检查分解是否成功（处理数值问题）
    if (ldlt.info() != Eigen::Success) {
        // 添加微小扰动确保正定（可选，根据实际场景调整）
        KXX += Eigen::MatrixXd::Identity(n, n) * 1e-8;
        ldlt.compute(KXX);
    }
    gtsam::Matrix KXX_inv = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
    
    // 对每个插值时间点进行估计
    for (double t : interpolatedTimes) {
        Vector12 state = gpPosteriorEstimate(t, originalTimes, originalTrajectory, KXX_inv);
        interpolatedTrajectory.push_back(state);
    }
    
    return interpolatedTrajectory;
}

// 设置插值因子（保持原有接口）
void GPInterpolator::setInterpolationFactor(int factor) {
    interpolationFactor_ = factor;
    interpolatedDt_ = dt_ / interpolationFactor_;
}

// 获取插值因子（保持原有接口）
int GPInterpolator::getInterpolationFactor() const {
    return interpolationFactor_;
}

// 获取插值后时间间隔（保持原有接口）
double GPInterpolator::getInterpolatedDt() const {
    return interpolatedDt_;
}

// 设置核函数特征长度（新增接口，控制平滑度）
void GPInterpolator::setLengthScale(double l) {
    lengthScale_ = l;
    inv_lengthScale_sq_ = 1.0 / (lengthScale_ * lengthScale_);  // 更新
}

// 设置观测噪声（新增接口，控制与原始点的接近程度）
void GPInterpolator::setNoiseSigma(double sigma) {
    noiseSigma_ = sigma;
}
    