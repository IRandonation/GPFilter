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
    noiseSigma_ = 0.00;  // 默认观测噪声较小，保证与原始轨迹接近
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
Vector6 GPInterpolator::gpPosteriorEstimate(
    double t,
    const std::vector<double>& originalTimes,
    const std::vector<Vector6>& originalStates,
    const gtsam::Matrix& KXX_inv) const {
    
    int n = originalTimes.size();
    
    // 计算当前时间t与所有原始时间点的协方差向量
    gtsam::Vector k(n);
    for (int i = 0; i < n; ++i) {
        k(i) = kernel(t, originalTimes[i]);
    }
    
    // 计算权重向量：K(x,X) * K(X,X)^{-1}
    gtsam::Vector weights = KXX_inv * k;
    
    // 对状态的6个维度（x, y, z, roll, pitch, yaw）分别进行估计
    Vector6 result = Vector6::Zero(6);
    for (int dim = 0; dim < 6; ++dim) {
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
std::vector<Vector6> GPInterpolator::interpolate(
    const std::vector<Vector6>& originalTrajectory) {
    
    std::vector<Vector6> interpolatedTrajectory;
    if (originalTrajectory.empty()) {
        return interpolatedTrajectory;
    }

    // 原始轨迹时间戳
    std::vector<double> originalTimes;
    for (size_t i = 0; i < originalTrajectory.size(); ++i) {
        originalTimes.push_back(i * dt_);
    }

    // 插值时间戳：插值点 + 原始点（保证端点精确对齐）
    std::vector<double> interpolatedTimes;
    double totalTime = (originalTrajectory.size() - 1) * dt_;
    for (double t = 0; t <= totalTime + 1e-12; t += interpolatedDt_) {
        interpolatedTimes.push_back(t);
    }
    interpolatedTimes.insert(interpolatedTimes.end(), originalTimes.begin(), originalTimes.end());
    std::sort(interpolatedTimes.begin(), interpolatedTimes.end());
    interpolatedTimes.erase(std::unique(interpolatedTimes.begin(), interpolatedTimes.end()), interpolatedTimes.end());

    // 构建协方差矩阵
    int n = originalTimes.size();
    gtsam::Matrix KXX = buildCovarianceMatrix(originalTimes);

    // Cholesky分解
    Eigen::LDLT<gtsam::Matrix> ldlt(KXX);
    if (ldlt.info() != Eigen::Success) {
        KXX += Eigen::MatrixXd::Identity(n, n) * 1e-10;
        ldlt.compute(KXX);
    }

    // 预构建每个维度的观测向量 y[dim][i]
    std::vector<gtsam::Vector> ys(6, gtsam::Vector(n));
    for (int dim = 0; dim < 6; ++dim) {
        for (int i = 0; i < n; ++i) {
            ys[dim](i) = originalTrajectory[i](dim);
        }
    }

    // 插值
    interpolatedTrajectory.reserve(interpolatedTimes.size());
    for (double t : interpolatedTimes) {
        // 协方差向量 k
        gtsam::Vector k(n);
        for (int i = 0; i < n; ++i) {
            k(i) = kernel(t, originalTimes[i]);
        }

        // 解方程 (更稳定，不显式求逆)
        gtsam::Vector weights = ldlt.solve(k);

        // 多维度插值
        Vector6 state = Vector6::Zero(6);
        for (int dim = 0; dim < 6; ++dim) {
            state(dim) = weights.dot(ys[dim]);
        }

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
    