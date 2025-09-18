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
    double P[3][3] {{1,0,0},{0,1,0},{0,0,1}}; // covariance

    // Noise
    double Q[3][3] {{1e-4,0,0},{0,1e-3,0},{0,0,1e-2}}; // process
    double R {1e-3}; // measurement (position)

    double Ts {0.01};

    void setNoise(double qpos, double qvel, double qacc, double r) {
        Q[0][0] = qpos; Q[1][1] = qvel; Q[2][2] = qacc; R = r;
    }

    void setDt(double ts) { Ts = ts; }

    // Predict step (constant acceleration)
    void predict() {
        // x = A x
        const double dt = Ts;
        double x_pred[3];
        x_pred[0] = x[0] + dt * x[1] + 0.5 * dt * dt * x[2];
        x_pred[1] = x[1] + dt * x[2];
        x_pred[2] = x[2];

        // A matrix
        double A[3][3] = {
            {1, dt, 0.5*dt*dt},
            {0,  1,        dt},
            {0,  0,         1}
        };

        // P = A P A^T + Q
        double AP[3][3]{};
        for (int i=0;i<3;++i) {
            for (int j=0;j<3;++j) {
                AP[i][j] = 0.0;
                for (int k=0;k<3;++k) AP[i][j] += A[i][k]*P[k][j];
            }
        }
        double At[3][3];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) At[i][j] = A[j][i];
        double APAt[3][3]{};
        for (int i=0;i<3;++i) {
            for (int j=0;j<3;++j) {
                APAt[i][j]=0.0;
                for (int k=0;k<3;++k) APAt[i][j]+=AP[i][k]*At[k][j];
                APAt[i][j]+=Q[i][j];
            }
        }
        // commit
        for (int i=0;i<3;++i) x[i]=x_pred[i];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) P[i][j]=APAt[i][j];
    }

    // Update with position-only measurement z
    void update(double z) {
        // H = [1 0 0]
        double y = z - x[0];
        double S = P[0][0] + R; // HPH^T + R
        double K[3];
        K[0] = P[0][0]/S; // first column because H^T = [1;0;0]
        K[1] = P[1][0]/S;
        K[2] = P[2][0]/S;

        // x = x + K y
        x[0] += K[0]*y;
        x[1] += K[1]*y;
        x[2] += K[2]*y;

        // P = (I - K H) P => rank-1 update
        double P_new[3][3];
        for (int i=0;i<3;++i) {
            for (int j=0;j<3;++j) {
                double IminusKH = (i==j?1.0:0.0) - K[i]* (j==0?1.0:0.0);
                double sum = 0.0;
                for (int k=0;k<3;++k) sum += IminusKH * P[i][j];
                P_new[i][j] = P[i][j] - K[i]*P[0][j];
            }
        }
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) P[i][j]=P_new[i][j];
    }
};

struct Vec3 { double x, y, z; };

static bool parse_args(int argc, char** argv,
                       std::string& csv_path, std::string& out_path,
                       double& dt, double& qpos, double& qvel, double& qacc, double& r)
{
    // defaults
    csv_path = "../../data/trajectory.csv";
    out_path = ""; // empty => no file write
    dt = 0.01; qpos = 1e-4; qvel = 1e-3; qacc = 1e-2; r = 1e-3;

    for (int i=1;i<argc;++i) {
        std::string arg = argv[i];
        auto next_val = [&](double& v){ if (i+1<argc) { v = std::stod(argv[++i]); return true; } return false; };
        auto next_str = [&](std::string& s){ if (i+1<argc) { s = argv[++i]; return true; } return false; };
        if (arg == "--csv") next_str(csv_path);
        else if (arg == "--out") next_str(out_path);
        else if (arg == "--dt") next_val(dt);
        else if (arg == "--q_pos") next_val(qpos);
        else if (arg == "--q_vel") next_val(qvel);
        else if (arg == "--q_acc") next_val(qacc);
        else if (arg == "--r") next_val(r);
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./gpfilter_kf --csv <path> --dt <sec> --q_pos <v> --q_vel <v> --q_acc <v> --r <v> --out <path>\n";
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            return false;
        }
    }
    return true;
}

static bool read_xyz_csv(const std::string& path, std::vector<Vec3>& out) {
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
        if (vals.size() >= 3) {
            out.push_back({vals[0], vals[1], vals[2]});
        }
    }
    return true;
}

static void write_xyz_out(const std::string& path,
                          const std::vector<Vec3>& raw,
                          const std::vector<Vec3>& pos,
                          const std::vector<Vec3>& vel,
                          const std::vector<Vec3>& acc)
{
    std::ofstream ofs(path);
    ofs << "raw_x,raw_y,raw_z,px,py,pz,vx,vy,vz,ax,ay,az\n";
    for (size_t i=0;i<pos.size();++i) {
        ofs << raw[i].x << "," << raw[i].y << "," << raw[i].z << ","
            << pos[i].x << "," << pos[i].y << "," << pos[i].z << ","
            << vel[i].x << "," << vel[i].y << "," << vel[i].z << ","
            << acc[i].x << "," << acc[i].y << "," << acc[i].z << "\n";
    }
}

int main(int argc, char** argv) {
    std::string csv_path, out_path;
    double dt, qpos, qvel, qacc, r;
    if (!parse_args(argc, argv, csv_path, out_path, dt, qpos, qvel, qacc, r)) return 1;

    std::vector<Vec3> raw; raw.reserve(10000);
    if (!read_xyz_csv(csv_path, raw)) return 2;
    if (raw.empty()) { std::cerr << "No data found in CSV\n"; return 3; }

    // Three independent filters for x, y, z
    KF1DCA kfx, kfy, kfz;
    kfx.setDt(dt); kfy.setDt(dt); kfz.setDt(dt);
    kfx.setNoise(qpos,qvel,qacc,r);
    kfy.setNoise(qpos,qvel,qacc,r);
    kfz.setNoise(qpos,qvel,qacc,r);

    std::vector<Vec3> pos, vel, acc;
    pos.reserve(raw.size()); vel.reserve(raw.size()); acc.reserve(raw.size());

    for (const auto& m : raw) {
        // predict
        kfx.predict(); kfy.predict(); kfz.predict();
        // update
        kfx.update(m.x); kfy.update(m.y); kfz.update(m.z);
        // collect
        pos.push_back({kfx.x[0], kfy.x[0], kfz.x[0]});
        vel.push_back({kfx.x[1], kfy.x[1], kfz.x[1]});
        acc.push_back({kfx.x[2], kfy.x[2], kfz.x[2]});
    }

    if (!out_path.empty()) {
        write_xyz_out(out_path, raw, pos, vel, acc);
        std::cout << "Saved: " << out_path << " (" << pos.size() << " rows)\n";
    } else {
        std::cout << "Filtered samples: " << pos.size() << "\n";
        std::cout << "Last state (x): p=" << kfx.x[0] << ", v=" << kfx.x[1] << ", a=" << kfx.x[2] << "\n";
    }
    return 0;
}