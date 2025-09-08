// GPInterpolator.h
#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <vector>
#include <cmath>

/**
 * GPInterpolator类实现高斯过程插值功能
 * 用于将原始轨迹数据点插值为更高密度的轨迹点，以体现平滑性
 */
class GPInterpolator {
public:
    /**
     * 构造函数
     * @param dt 原始时间步长
     * @param interpolationFactor 插值因子（默认为10，表示将原始数据点插值为10倍）
     */
    GPInterpolator(double dt, int interpolationFactor = 10);
    
    /**
     * 对轨迹进行插值
     * @param originalTrajectory 原始轨迹数据
     * @return 插值后的轨迹数据
     */
    std::vector<gtsam::Vector4> interpolate(const std::vector<gtsam::Vector4>& originalTrajectory);
    
    /**
     * 计算插值点
     * @param state1 前一个状态点
     * @param state2 后一个状态点
     * @param alpha 插值参数（0到1之间）
     * @return 插值后的状态点
     */
    gtsam::Vector4 interpolatePoint(const gtsam::Vector4& state1, const gtsam::Vector4& state2, double alpha);
    
    /**
     * 设置插值因子
     * @param factor 新的插值因子
     */
    void setInterpolationFactor(int factor);
    
    /**
     * 获取插值因子
     * @return 当前插值因子
     */
    int getInterpolationFactor() const;
    
    /**
     * 获取插值后的时间步长
     * @return 插值后的时间步长
     */
    double getInterpolatedDt() const;

private:
    double dt_;                    // 原始时间步长
    int interpolationFactor_;      // 插值因子
    double interpolatedDt_;        // 插值后的时间步长
    
    /**
     * 计算状态转移矩阵
     * @param dt 时间步长
     * @return 状态转移矩阵
     */
    gtsam::Matrix4 computeStateTransitionMatrix(double dt) const;
    
    /**
     * 计算过程噪声协方差矩阵
     * @param dt 时间步长
     * @return 过程噪声协方差矩阵
     */
    gtsam::Matrix4 computeProcessNoiseCovariance(double dt) const;
    
    /**
     * 高斯过程回归插值
     * @param state1 前一个状态点
     * @param state2 后一个状态点
     * @param alpha 插值参数
     * @return 插值后的状态点
     */
    gtsam::Vector4 gpRegressionInterpolation(const gtsam::Vector4& state1, 
                                            const gtsam::Vector4& state2, 
                                            double alpha);
};