#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct KF1DCA {
    // State: [pos, vel, acc]
    double x[3] {0,0,0};
    double P[3][3] {{1,0,0},{0,10,0},{0,0,100}}; // covariance - higher uncertainty for vel/acc
    double x_pred[3];
    // Noise
    double Q[3][3] {{1e-1,0,0},{0,1e-1,0},{0,0,1e-1}}; // process
    double R {1e-3}; // measurement (position)

    double Ts {0.01};

    // RTS Smoother storage
    std::vector<std::vector<double>> x_history;     // filtered states [N][3]
    std::vector<std::vector<double>> x_pred_history; // predicted states [N][3]
    std::vector<std::vector<std::vector<double>>> P_history;   // filtered covariances [N][3][3]
    std::vector<std::vector<std::vector<double>>> P_pred_history; // predicted covariances [N][3][3]
    std::vector<std::vector<double>> x_smooth;      // smoothed states [N][3]
    bool enable_smoothing {true};

    void setNoise(double qpos, double qvel, double qacc, double r) {
        Q[0][0] = qpos; Q[1][1] = qvel; Q[2][2] = qacc; R = r;
    }

    void setDt(double ts) { Ts = ts; }

    // Predict step (constant acceleration)
    void predict() {
        const double dt = Ts;
        
        // State prediction: x_pred = A * x
        x_pred[0] = x[0] + dt * x[1] + 0.5 * dt * dt * x[2];
        x_pred[1] = x[1] + dt * x[2];
        x_pred[2] = x[2];

        // A matrix for constant acceleration model
        double A[3][3] = {
            {1, dt, 0.5*dt*dt},
            {0,  1,        dt},
            {0,  0,         1}
        };

        // Covariance prediction: P_pred = A * P * A^T + Q
        double AP[3][3] = {0};
        for (int i=0;i<3;++i) {
            for (int j=0;j<3;++j) {
                for (int k=0;k<3;++k) {
                    AP[i][j] += A[i][k] * P[k][j];
                }
            }
        }
        
        double APAt[3][3] = {0};
        for (int i=0;i<3;++i) {
            for (int j=0;j<3;++j) {
                for (int k=0;k<3;++k) {
                    APAt[i][j] += AP[i][k] * A[j][k]; // A[j][k] is A^T[k][j]
                }
                APAt[i][j] += Q[i][j];
            }
        }
        
        // Store predicted state and covariance for RTS smoothing
        if (enable_smoothing) {
            x_pred_history.push_back(std::vector<double>(3));
            P_pred_history.push_back(std::vector<std::vector<double>>(3, std::vector<double>(3)));
            
            for (int i=0;i<3;++i) {
                x_pred_history.back()[i] = x_pred[i];
                for (int j=0;j<3;++j) {
                    P_pred_history.back()[i][j] = APAt[i][j];
                }
            }
        }
        
        // Update only covariance in predict step (state updated in update step)
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) P[i][j] = APAt[i][j];
    }

    // Update with position-only measurement z
    void update(double z) {
        // H = [1 0 0]
        double y = z - x_pred[0];
        double S = P[0][0] + R; // HPH^T + R
        double K[3];
        K[0] = P[0][0]/S; // first column because H^T = [1;0;0]
        K[1] = P[1][0]/S;
        K[2] = P[2][0]/S;

        // x = x + K y
        x[0] = x_pred[0] + K[0]*y;
        x[1] = x_pred[1] + K[1]*y;
        x[2] = x_pred[2] + K[2]*y;

        // P = (I - K H) P_pred => correct implementation using P_pred from predict step
        // H = [1 0 0], so (I - K H) = [[1-K[0], 0, 0], [-K[1], 1, 0], [-K[2], 0, 1]]
        double P_old[3][3];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) P_old[i][j] = P[i][j];
        
        P[0][0] = (1.0 - K[0]) * P_old[0][0];
        P[0][1] = (1.0 - K[0]) * P_old[0][1];
        P[0][2] = (1.0 - K[0]) * P_old[0][2];
        P[1][0] = -K[1] * P_old[0][0] + P_old[1][0];
        P[1][1] = -K[1] * P_old[0][1] + P_old[1][1];
        P[1][2] = -K[1] * P_old[0][2] + P_old[1][2];
        P[2][0] = -K[2] * P_old[0][0] + P_old[2][0];
        P[2][1] = -K[2] * P_old[0][1] + P_old[2][1];
        P[2][2] = -K[2] * P_old[0][2] + P_old[2][2];

        // Store filtered state and covariance for RTS smoothing
        if (enable_smoothing) {
            x_history.push_back(std::vector<double>(3));
            P_history.push_back(std::vector<std::vector<double>>(3, std::vector<double>(3)));
            
            for (int i=0;i<3;++i) {
                x_history.back()[i] = x[i];
                for (int j=0;j<3;++j) {
                    P_history.back()[i][j] = P[i][j];
                }
            }
        }
     }

     // RTS Smoother implementation
     void smooth() {
         if (!enable_smoothing || x_history.empty()) return;
         
         size_t N = x_history.size();
         x_smooth.resize(N);
         for (size_t k = 0; k < N; ++k) {
             x_smooth[k].resize(3);
         }
         
         // Initialize with last filtered state
         for (int i=0;i<3;++i) {
             x_smooth[N-1][i] = x_history[N-1][i];
         }
         
         const double dt = Ts;
         // A matrix for constant acceleration model
         double A[3][3] = {
             {1, dt, 0.5*dt*dt},
             {0,  1,        dt},
             {0,  0,         1}
         };
         
         // Backward pass
         for (int k = N-2; k >= 0; --k) {
             // Compute smoother gain: C_k = P_k * A^T * P_pred_{k+1}^{-1}
             double C[3][3] = {0};
             
             // Compute P_pred_{k+1}^{-1}
             double P_pred_inv[3][3];
             double det = P_pred_history[k+1][0][0] * (P_pred_history[k+1][1][1] * P_pred_history[k+1][2][2] - P_pred_history[k+1][1][2] * P_pred_history[k+1][2][1])
                        - P_pred_history[k+1][0][1] * (P_pred_history[k+1][1][0] * P_pred_history[k+1][2][2] - P_pred_history[k+1][1][2] * P_pred_history[k+1][2][0])
                        + P_pred_history[k+1][0][2] * (P_pred_history[k+1][1][0] * P_pred_history[k+1][2][1] - P_pred_history[k+1][1][1] * P_pred_history[k+1][2][0]);
             
             if (std::abs(det) > 1e-12) {
                 P_pred_inv[0][0] = (P_pred_history[k+1][1][1] * P_pred_history[k+1][2][2] - P_pred_history[k+1][1][2] * P_pred_history[k+1][2][1]) / det;
                 P_pred_inv[0][1] = -(P_pred_history[k+1][0][1] * P_pred_history[k+1][2][2] - P_pred_history[k+1][0][2] * P_pred_history[k+1][2][1]) / det;
                 P_pred_inv[0][2] = (P_pred_history[k+1][0][1] * P_pred_history[k+1][1][2] - P_pred_history[k+1][0][2] * P_pred_history[k+1][1][1]) / det;
                 P_pred_inv[1][0] = -(P_pred_history[k+1][1][0] * P_pred_history[k+1][2][2] - P_pred_history[k+1][1][2] * P_pred_history[k+1][2][0]) / det;
                 P_pred_inv[1][1] = (P_pred_history[k+1][0][0] * P_pred_history[k+1][2][2] - P_pred_history[k+1][0][2] * P_pred_history[k+1][2][0]) / det;
                 P_pred_inv[1][2] = -(P_pred_history[k+1][0][0] * P_pred_history[k+1][1][2] - P_pred_history[k+1][0][2] * P_pred_history[k+1][1][0]) / det;
                 P_pred_inv[2][0] = (P_pred_history[k+1][1][0] * P_pred_history[k+1][2][1] - P_pred_history[k+1][1][1] * P_pred_history[k+1][2][0]) / det;
                 P_pred_inv[2][1] = -(P_pred_history[k+1][0][0] * P_pred_history[k+1][2][1] - P_pred_history[k+1][0][1] * P_pred_history[k+1][2][0]) / det;
                 P_pred_inv[2][2] = (P_pred_history[k+1][0][0] * P_pred_history[k+1][1][1] - P_pred_history[k+1][0][1] * P_pred_history[k+1][1][0]) / det;
                 
                 // C_k = P_k * A^T * P_pred_{k+1}^{-1}
                 double PA[3][3] = {0};
                 for (int i=0;i<3;++i) {
                     for (int j=0;j<3;++j) {
                         for (int l=0;l<3;++l) {
                             PA[i][j] += P_history[k][i][l] * A[j][l]; // A[j][l] is A^T[l][j]
                         }
                     }
                 }
                 
                 for (int i=0;i<3;++i) {
                     for (int j=0;j<3;++j) {
                         for (int l=0;l<3;++l) {
                             C[i][j] += PA[i][l] * P_pred_inv[l][j];
                         }
                     }
                 }
                 
                 // Smoothed state: x_smooth_k = x_k + C_k * (x_smooth_{k+1} - x_pred_{k+1})
                 double diff[3];
                 for (int i=0;i<3;++i) {
                     diff[i] = x_smooth[k+1][i] - x_pred_history[k+1][i];
                 }
                 
                 for (int i=0;i<3;++i) {
                     x_smooth[k][i] = x_history[k][i];
                     for (int j=0;j<3;++j) {
                         x_smooth[k][i] += C[i][j] * diff[j];
                     }
                 }
             } else {
                 // Fallback to filtered estimate if matrix is singular
                 for (int i=0;i<3;++i) {
                     x_smooth[k][i] = x_history[k][i];
                 }
             }
         }
     }

     // Get smoothed state at index k
     void getSmoothedState(size_t k, double state[3]) {
         if (k < x_smooth.size() && x_smooth[k].size() == 3) {
             for (int i=0;i<3;++i) {
                 state[i] = x_smooth[k][i];
             }
         } else {
             for (int i=0;i<3;++i) {
                 state[i] = x[i];
             }
         }
     }

     // Clear history for new sequence
     void clearHistory() {
         x_history.clear();
         x_pred_history.clear();
         P_history.clear();
         P_pred_history.clear();
         x_smooth.clear();
     }
};

struct Vec3 { double x, y, z; };
struct Vec6 { double x, y, z, rx, ry, rz; };
struct Vec9 { double x, y, z, vx, vy, vz, ax, ay, az; };

static bool read_6d_csv(const std::string& path, std::vector<Vec6>& out) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Failed to open CSV: " << path << "\n";
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<double> vals;
        while (std::getline(ss, item, ',')) {
            try {
                vals.push_back(std::stod(item));
            } catch (...) { vals.push_back(0.0); }
        }
        if (vals.size() >= 6) {
            out.push_back({vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]});
        }
    }
    return true;
}

static void write_full_state_out(const std::string& path,
                                 const std::vector<Vec9>& pos_state,
                                 const std::vector<Vec9>& rpy_state)
{
    std::ofstream ofs(path);
    ofs << "x,y,z,vx,vy,vz,ax,ay,az,rx,ry,rz,vrx,vry,vrz,arx,ary,arz\n";
    for (size_t i=0;i<pos_state.size();++i) {
        ofs << pos_state[i].x << "," << pos_state[i].y << "," << pos_state[i].z << ","
            << pos_state[i].vx << "," << pos_state[i].vy << "," << pos_state[i].vz << ","
            << pos_state[i].ax << "," << pos_state[i].ay << "," << pos_state[i].az << ","
            << rpy_state[i].x << "," << rpy_state[i].y << "," << rpy_state[i].z << ","
            << rpy_state[i].vx << "," << rpy_state[i].vy << "," << rpy_state[i].vz << ","
            << rpy_state[i].ax << "," << rpy_state[i].ay << "," << rpy_state[i].az << "\n";
    }
}



int main(int argc, char** argv) {
    std::string csv_path = "../../data/trajectory.csv", out_path = "../../output/smoothed_trajectory.csv";
    // 调整参数以获得更平滑的速度和加速度
    // 减小 qvel 和 qacc，增大 r
    double dt = 1.0/60.0, qpos = 1e-5, qvel = 1e-3, qacc = 1e-2, r = 1e-4; 
    // if (!parse_args(argc, argv, csv_path, out_path, dt, qpos, qvel, qacc, r)) return 1;

    std::vector<Vec6> raw; raw.reserve(10000);
    if (!read_6d_csv(csv_path, raw)) return 2;
    if (raw.empty()) { std::cerr << "No data found in CSV\n"; return 3; }

    // Six independent filters for x, y, z, rx, ry, rz
    KF1DCA kfx, kfy, kfz, kfrx, kfry, kfrz;
    kfx.setDt(dt); kfy.setDt(dt); kfz.setDt(dt);
    kfrx.setDt(dt); kfry.setDt(dt); kfrz.setDt(dt);
    kfx.setNoise(qpos,qvel,qacc,r);
    kfy.setNoise(qpos,qvel,qacc,r);
    kfz.setNoise(qpos,qvel,qacc,r);
    kfrx.setNoise(qpos,qvel,qacc,r);
    kfry.setNoise(qpos,qvel,qacc,r);
    kfrz.setNoise(qpos,qvel,qacc,r);

    // Initialize Kalman filter states with the first measurement
    kfx.x[0] = raw[0].x;
    kfy.x[0] = raw[0].y;
    kfz.x[0] = raw[0].z;
    kfrx.x[0] = raw[0].rx;
    kfry.x[0] = raw[0].ry;
    kfrz.x[0] = raw[0].rz;

    std::vector<Vec9> pos_state, rpy_state;
    pos_state.reserve(raw.size()); rpy_state.reserve(raw.size());

    // Phase 1: Forward Kalman filtering
    for (const auto& m : raw) {
        // predict
        kfx.predict(); kfy.predict(); kfz.predict();
        kfrx.predict(); kfry.predict(); kfrz.predict();
        // update
        kfx.update(m.x); kfy.update(m.y); kfz.update(m.z);
        kfrx.update(m.rx); kfry.update(m.ry); kfrz.update(m.rz);
    }

    // Phase 2: RTS Smoothing
    kfx.smooth(); kfy.smooth(); kfz.smooth();
    kfrx.smooth(); kfry.smooth(); kfrz.smooth();

    // Phase 3: Collect smoothed states
    for (size_t i = 0; i < raw.size(); ++i) {
        double temp_state[3];
        
        // Get smoothed states for position (x, y, z)
        kfx.getSmoothedState(i, temp_state);
        double pos_x = temp_state[0], vel_x = temp_state[1], acc_x = temp_state[2];
        
        kfy.getSmoothedState(i, temp_state);
        double pos_y = temp_state[0], vel_y = temp_state[1], acc_y = temp_state[2];
        
        kfz.getSmoothedState(i, temp_state);
        double pos_z = temp_state[0], vel_z = temp_state[1], acc_z = temp_state[2];
        
        // Get smoothed states for orientation (rx, ry, rz)
        kfrx.getSmoothedState(i, temp_state);
        double rpy_x = temp_state[0], vel_rx = temp_state[1], acc_rx = temp_state[2];
        
        kfry.getSmoothedState(i, temp_state);
        double rpy_y = temp_state[0], vel_ry = temp_state[1], acc_ry = temp_state[2];
        
        kfrz.getSmoothedState(i, temp_state);
        double rpy_z = temp_state[0], vel_rz = temp_state[1], acc_rz = temp_state[2];
        
        // Store smoothed complete state
        pos_state.push_back({pos_x, pos_y, pos_z,           // position
                            vel_x, vel_y, vel_z,            // velocity
                            acc_x, acc_y, acc_z});          // acceleration
        rpy_state.push_back({rpy_x, rpy_y, rpy_z,           // orientation
                            vel_rx, vel_ry, vel_rz,         // angular velocity
                            acc_rx, acc_ry, acc_rz});       // angular acceleration
    }

    if (!out_path.empty()) {
        write_full_state_out(out_path, pos_state, rpy_state);
        
        std::cout << "Saved: " << out_path << " (" << pos_state.size() << " rows)\n";
        std::cout << "Output format: x,y,z,vx,vy,vz,ax,ay,az,rx,ry,rz,vrx,vry,vrz,arx,ary,arz\n";
    } else {
        std::cout << "Filtered samples: " << pos_state.size() << "\n";
        std::cout << "Last state (x): p=" << kfx.x[0] << ", v=" << kfx.x[1] << ", a=" << kfx.x[2] << "\n";
        std::cout << "Last state (rx): p=" << kfrx.x[0] << ", v=" << kfrx.x[1] << ", a=" << kfrx.x[2] << "\n";
    }
    return 0;
}