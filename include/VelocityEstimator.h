// include/VelocityEstimator.h
#pragma once

#include <vector>
#include <gtsam/geometry/Point2.h>
#include <gtsam/base/Vector.h>

/**
 * 速度估计器类
 * 提供多种速度估计方法，包括基于匀速模型和卡尔曼滤波的估计
 */
class VelocityEstimator {
public:
    /**
     * 枚举类型：定义不同的速度估计方法
     */
    enum EstimationMethod {
        SIMPLE_DIFFERENCE,  // 简单差分法（原始方法）
        CONSTANT_VELOCITY,  // 匀速模型估计
        KALMAN_FILTER      // 卡尔曼滤波估计
    };

    /**
     * 构造函数
     * @param method 速度估计方法
     * @param windowSize 用于估计的窗口大小（多帧数据）
     * @param dt 时间步长
     */
    VelocityEstimator(EstimationMethod method = CONSTANT_VELOCITY, 
                     int windowSize = 5, 
                     double dt = 0.1);

    /**
     * 估计整个轨迹的速度
     * @param positions 位置点序列
     * @return 包含位置和估计速度的状态向量序列
     */
    std::vector<gtsam::Vector4> estimateVelocities(
        const std::vector<gtsam::Point2>& positions);

    /**
     * 设置估计方法
     * @param method 速度估计方法
     */
    void setEstimationMethod(EstimationMethod method);

    /**
     * 设置窗口大小
     * @param windowSize 窗口大小
     */
    void setWindowSize(int windowSize);

    /**
     * 设置时间步长
     * @param dt 时间步长
     */
    void setTimeStep(double dt);

    /**
     * 获取当前估计方法
     * @return 估计方法
     */
    EstimationMethod getEstimationMethod() const;

    /**
     * 获取窗口大小
     * @return 窗口大小
     */
    int getWindowSize() const;

    /**
     * 获取时间步长
     * @return 时间步长
     */
    double getTimeStep() const;

private:
    EstimationMethod method_;    // 估计方法
    int windowSize_;             // 窗口大小
    double dt_;                  // 时间步长

    /**
     * 简单差分法估计速度
     * @param positions 位置点序列
     * @return 包含位置和估计速度的状态向量序列
     */
    std::vector<gtsam::Vector4> estimateWithSimpleDifference(
        const std::vector<gtsam::Point2>& positions);

    /**
     * 基于匀速模型估计速度
     * @param positions 位置点序列
     * @return 包含位置和估计速度的状态向量序列
     */
    std::vector<gtsam::Vector4> estimateWithConstantVelocity(
        const std::vector<gtsam::Point2>& positions);

    /**
     * 基于卡尔曼滤波估计速度
     * @param positions 位置点序列
     * @return 包含位置和估计速度的状态向量序列
     */
    std::vector<gtsam::Vector4> estimateWithKalmanFilter(
        const std::vector<gtsam::Point2>& positions);

    /**
     * 使用最小二乘法拟合局部匀速模型
     * @param positions 局部位置点序列
     * @param startIndex 起始索引
     * @param endIndex 结束索引
     * @return 估计的速度 (vx, vy)
     */
    gtsam::Vector2 fitLocalConstantVelocity(
        const std::vector<gtsam::Point2>& positions,
        int startIndex, 
        int endIndex);

    /**
     * 计算加权平均速度，考虑距离权重
     * @param velocities 速度序列
     * @param weights 权重序列
     * @return 加权平均速度
     */
    gtsam::Vector2 computeWeightedAverageVelocity(
        const std::vector<gtsam::Vector2>& velocities,
        const std::vector<double>& weights);
};