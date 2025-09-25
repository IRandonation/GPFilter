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
    double dt = 1.0/60.0, qpos = 1e-5, qvel = 1e-4, qacc = 1e-3, r = 1e-3; 
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

    for (const auto& m : raw) {
        // predict
        kfx.predict(); kfy.predict(); kfz.predict();
        kfrx.predict(); kfry.predict(); kfrz.predict();
        // update
        kfx.update(m.x); kfy.update(m.y); kfz.update(m.z);
        kfrx.update(m.rx); kfry.update(m.ry); kfrz.update(m.rz);
        // collect complete state: position, velocity, acceleration
        pos_state.push_back({kfx.x[0], kfy.x[0], kfz.x[0],     // position
                            kfx.x[1], kfy.x[1], kfz.x[1],     // velocity
                            kfx.x[2], kfy.x[2], kfz.x[2]});   // acceleration
        rpy_state.push_back({kfrx.x[0], kfry.x[0], kfrz.x[0], // orientation
                            kfrx.x[1], kfry.x[1], kfrz.x[1],  // angular velocity
                            kfrx.x[2], kfry.x[2], kfrz.x[2]}); // angular acceleration
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