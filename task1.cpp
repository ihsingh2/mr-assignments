#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct CameraIntrinsics {
    double fx;
    double fy;
    double cx;
    double cy;
};

struct CameraPose {
    double rotation[3];
    double translation[3];

    CameraPose() {}
    CameraPose(double r0, double r1, double r2, double t0, double t1, double t2) {
        rotation[0] = r0;
        rotation[1] = r1;
        rotation[2] = r2;
        translation[0] = t0;
        translation[1] = t1;
        translation[2] = t2;
    }

    std::array<std::array<double, 4>, 4> TransformationMatrix() const {
        double rotation_matrix[9];
        ceres::AngleAxisToRotationMatrix(rotation, rotation_matrix);

        std::array<std::array<double, 4>, 4> transformation_matrix = {0.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                transformation_matrix[j][i] = rotation_matrix[i * 3 + j];
            transformation_matrix[i][3] = translation[i];
        }
        transformation_matrix[3][3] = 1.0;

        return transformation_matrix;
    }
};

struct Correspondence {
    double px;
    double py;
    double X;
    double Y;
    double Z;
};

struct ReprojectionError {
    ReprojectionError(double observed_px, double observed_py, double X, double Y, double Z, const CameraIntrinsics& intrinsics)
        : observed_px_(observed_px), observed_py_(observed_py), X_(X), Y_(Y), Z_(Z), intrinsics_(intrinsics) {}

    template <typename T>
    bool operator()(const T* const rotation,
                    const T* const translation,
                    T* residuals) const {

        // Original point
        T p[3] = { T(X_), T(Y_), T(Z_) };

        // Apply rotation
        T p_rotated[3];
        ceres::AngleAxisRotatePoint(rotation, p, p_rotated);

        // Apply translation
        p_rotated[0] += translation[0];
        p_rotated[1] += translation[1];
        p_rotated[2] += translation[2];

        // Apply intrinsics
        T xp = T(intrinsics_.fx) * p_rotated[0] + T(intrinsics_.cx) * p_rotated[2];
        T yp = T(intrinsics_.fy) * p_rotated[1] + T(intrinsics_.cy) * p_rotated[2];

        // Normalization
        T u = xp / p_rotated[2];
        T v = yp / p_rotated[2];

        // Compute residuals
        residuals[0] = u - T(observed_px_);
        residuals[1] = v - T(observed_py_);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_px,
                                       const double observed_py,
                                       const double X,
                                       const double Y,
                                       const double Z,
                                       const CameraIntrinsics& intrinsics) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
            new ReprojectionError(observed_px, observed_py, X, Y, Z, intrinsics)));
    }

    double observed_px_;
    double observed_py_;
    double X_;
    double Y_;
    double Z_;
    CameraIntrinsics intrinsics_;
};

bool LoadCorrespondences(const std::string& filepath, std::vector<Correspondence>& correspondences) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open correspondence file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    while (getline(file, line)) {
        std::istringstream iss(line);
        Correspondence corr;
        int index; // to ignore
        if (!(iss >> corr.px >> corr.py >> corr.X >> corr.Y >> corr.Z >> index)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            return false;
        }
        correspondences.push_back(corr);
    }
    return true;
}

void InitializeCameraPoses(int num_poses, std::vector<CameraPose>& camera_poses) {
    camera_poses.insert(camera_poses.end(), {
        CameraPose(1, 0, 0, 0.0, 0.0, 263),
        CameraPose(-1, 0, 0, 0.0, 0.0, 289),
        CameraPose(1, 1, 0, 0.0, 0.0, 261),
        CameraPose(-1, -1, 1, 0.0, 0.0, 201),
        CameraPose(0, 0, 0.0, 0.0, 0.0, 298)
    });
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    int num_poses = 5;
    int image_width = 512;
    int image_height = 352;

    CameraIntrinsics intrinsics;
    intrinsics.fx = 721.5;
    intrinsics.fy = 721.5;
    intrinsics.cx = image_width / 2.0;
    intrinsics.cy = image_height / 2.0;

    std::vector<CameraPose> camera_poses;
    InitializeCameraPoses(num_poses, camera_poses);

    ceres::Problem problem;
    for (int i = 0; i < num_poses; ++i) {
        std::string filepath = "data/Correspondences/pose_" + std::to_string(i) + ".txt";
        std::vector<Correspondence> correspondences;

        if (!LoadCorrespondences(filepath, correspondences)) {
            std::cerr << "Failed to load correspondences for pose " << i << std::endl;
            return 1;
        }

        for (const auto& corr : correspondences) {
            ceres::CostFunction* cost_function = ReprojectionError::Create(corr.px, corr.py, corr.X, corr.Y, corr.Z, intrinsics);
            problem.AddResidualBlock(cost_function, nullptr, camera_poses[i].rotation, camera_poses[i].translation);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;
    options.function_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    for (int i = 0; i < num_poses; ++i) {
        std::cout << "Pose " << i << ":\n";
        std::array<std::array<double, 4>, 4> transform = camera_poses[i].TransformationMatrix();
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k)
                std::cout << transform[j][k] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
