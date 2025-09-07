// include/MeasurementFactor.h
#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Point2.h>

class MeasurementFactor : public gtsam::NoiseModelFactor1<gtsam::Vector4> {
public:
    gtsam::Point2 measured_;
    
    MeasurementFactor(gtsam::Key key, const gtsam::Point2& measured,
                    const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor1<gtsam::Vector4>(model, key), measured_(measured) {
    }
    
    gtsam::Vector evaluateError(const gtsam::Vector4& state,
                               gtsam::OptionalMatrixType H = nullptr) const override {
        if (H) {
            *H = (gtsam::Matrix(2,4) << 1,0,0,0,
                                      0,1,0,0).finished();
        }
        return gtsam::Vector2(state[0], state[1]) - measured_;
    }
};