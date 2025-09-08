// GPInterpolator.h
#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <vector>
#include <cmath>

class GPInterpolator {
public:
    GPInterpolator(double dt, int interpolationFactor = 10);
    
    std::vector<gtsam::Vector4> interpolate(const std::vector<gtsam::Vector4>& originalTrajectory);
    
    void setInterpolationFactor(int factor);
    int getInterpolationFactor() const;
    double getInterpolatedDt() const;

private:
    double dt_;
    int interpolationFactor_;
    double interpolatedDt_;
    
    gtsam::Matrix4 computeStateTransitionMatrix(double dt) const;
    gtsam::Matrix4 computeProcessNoiseCovariance(double dt) const;
    gtsam::Vector4 gpRegressionInterpolation(const gtsam::Vector4& state1,
                                            const gtsam::Vector4& state2,
                                            double alpha);
};