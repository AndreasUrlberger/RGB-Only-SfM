#include "BundleAdjuster.h"

#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/types.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>

#include "Eigen.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

class LoggingCallback : public ceres::IterationCallback
{
   public:
    explicit LoggingCallback(const std::string& log_file = "log_loss.csv")
        : log_file_(log_file), bundleAdjusterIteration_(0)
    {
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
    {
        losses_.push_back(summary.cost);
        return ceres::SOLVER_CONTINUE;
    }

    void SaveToFile()
    {
        std::ofstream ofs;
        ofs.open(log_file_, std::ofstream::out | std::ofstream::app);
        if (ofs.is_open())
        {
            size_t iteration = 0;
            for (const auto& loss : losses_)
            {
                ofs << bundleAdjusterIteration_ << "," << iteration++ << ","
                    << loss << "\n";
            }

            ofs.close();
        }
        losses_.clear();
    }
    std::string log_file_;
    size_t bundleAdjusterIteration_;

   private:
    std::vector<double> losses_;
};

// Define a to_string function for ceres::Jet
template <typename T, int N>
std::string to_string(const ceres::Jet<T, N>& jet)
{
    std::ostringstream oss;
    oss << "Jet(value: " << jet.a << ", derivatives: [";
    for (int i = 0; i < N; ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << jet.v[i];
    }
    oss << "])";
    return oss.str();
}

std::string to_string(double value) { return std::to_string(value); }

#define ENABLE_LOG_CERES

struct BundleAdjuster::impl
{
    struct SnavelyReprojectionError
    {
        SnavelyReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y)
        {
        }

        template <typename T>
        bool operator()(const T* const camera, const T* const point,
                        const T* const intrinsics, T* residuals) const
        {
            // camera [0,1,2] are angle-axis rotation
            T p[3];
            ceres::AngleAxisRotatePoint(camera, point, p);

            // camera[3,4,5] are the translation.
            p[0] += camera[3];
            p[1] += camera[4];
            p[2] += camera[5];

            // note: this varies from Ceres example reprojection error
            // in ceres example, radial distortion coefficients stored in 7-8
            const T& focal = intrinsics[0];
            const T& cx = intrinsics[1];
            const T& cy = intrinsics[2];

            T xp = ((focal * p[0]) + (cx * p[2])) / p[2];
            T yp = ((focal * p[1]) + (cy * p[2])) / p[2];

            // @TODO if support for radial distortion desired
            // Compute center of distortion. Sign change comes
            // from camera model that Snavely's bundler assumes,
            // whereby camera coordinate system has a negative z axis.
            // Apply second and fourth order radial distortion.
            // T r2 = xp * xp + yp * yp;
            T distortion = (T)1.0;  // 1.0 + r2 * (l1 + l2 * r2);

            // Compute final projected point position.
            T predicted_x = xp;
            T predicted_y = yp;

            // The error is the difference between the predicted and observed
            // position
            residuals[0] = predicted_x - observed_x;
            residuals[1] = predicted_y - observed_y;

            spdlog::trace("residual 0 predicted: {} observed: {} delta {}",
                          to_string(predicted_x), to_string(observed_x),
                          to_string(residuals[0]));
            spdlog::trace("residual 1 predicted: {} observed: {} delta {}",
                          to_string(predicted_y), to_string(observed_y),
                          to_string(residuals[1]));

            return true;
        }

        // Factory to hide the construction of the CostFunction object
        // from client code
        static ceres::CostFunction* Create(const double observed_x,
                                           const double observed_y)
        {
            return new ceres::AutoDiffCostFunction<
                BundleAdjuster::impl::SnavelyReprojectionError, 2, 6, 3, 3>(
                observed_x, observed_y);
        }

        double observed_x;
        double observed_y;
    };

    std::shared_ptr<ceres::Problem> problem_;
    std::shared_ptr<ceres::LossFunction> loss_function_;
    BundleAdjuster::LossFunctionType loss_function_type_;
    std::vector<ceres::ResidualBlockId> residual_ids_;

    std::vector<BundleAdjuster::Observation> observations;
    std::vector<double> cameras;
    std::vector<double> cameraIntrinsics;
    size_t cameraIdx;
    std::vector<double> worldPoints;
    size_t worldPointIdx;

    LoggingCallback loggingCallback_;
    std::string residualFile_;
    std::ofstream residualFileFs_;
};

BundleAdjuster::BundleAdjuster(std::string logFilePath, const std::string& lossFunctionType) : pImpl{std::make_unique<impl>()}
{
    pImpl->cameraIdx = 0;
    pImpl->worldPointIdx = 0;

    // pre-allocate storage to avoid dynamic allocations
    pImpl->cameras.resize(1000000);
    pImpl->worldPoints.resize(100'000'000);
    pImpl->cameraIntrinsics.resize(3);

    // log residual values to file for debug
    // @TODO: remove, likely slows down execution time
    auto now = std::chrono::system_clock::now();
    auto UTC =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
            .count();

    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");

    pImpl->loggingCallback_.log_file_ = logFilePath;
    pImpl->loggingCallback_.bundleAdjusterIteration_ = 0;
    pImpl->residualFile_ = "residual_file.csv";
    pImpl->residualFileFs_.open(pImpl->residualFile_,
                                std::ofstream::out | std::ofstream::app);

    // default
    pImpl->loss_function_type_ = BundleAdjuster::LossFunctionType::TRIVIAL;

    if (lossFunctionType == "CAUCHY")
    {
        pImpl->loss_function_type_ = BundleAdjuster::LossFunctionType::CAUCHY;
    }
    else if (lossFunctionType == "HUBER")
    {
        pImpl->loss_function_type_ = BundleAdjuster::LossFunctionType::HUBER;
    }
}

BundleAdjuster::~BundleAdjuster() { pImpl->residualFileFs_.close(); }

void BundleAdjuster::setupProblem()
{
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    pImpl->problem_ = std::make_shared<ceres::Problem>(problem_options);

    if (pImpl->loss_function_type_ == BundleAdjuster::LossFunctionType::CAUCHY)
    {
        spdlog::info("Using CAUCHY loss function with scale value 1.0");
        pImpl->loss_function_ = std::make_shared<ceres::CauchyLoss>(1.0);
    }
    else if (pImpl->loss_function_type_ == BundleAdjuster::LossFunctionType::HUBER)
    {
        spdlog::info("Using HUBER loss function with scale value 1.0");
        pImpl->loss_function_ = std::make_shared<ceres::HuberLoss>(1.0);
    }
    else if (pImpl->loss_function_type_ == BundleAdjuster::LossFunctionType::TRIVIAL)
    {
        spdlog::info("Using TRIVIAL loss function");
        pImpl->loss_function_ = std::make_shared<ceres::TrivialLoss>();
    }
}

void BundleAdjuster::solve()
{
    // given image feature loactions and correspondences,
    //  find 3D point positions and camera parameters that minimize reprojection
    //  error

    // non-linear least squares problem: error = squared L2 norm between
    // observed
    //  feature location and projection of corresponding 3D point on image plane
    //  of camera
    // 1. create loss function, given options
    // 2. setup problem
    // 3. setup ceres solver
    // 4. run solver
    // 5. print solver summary

    spdlog::info("in solve()");

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 150;
    options.max_linear_solver_iterations = 200;
    options.max_num_consecutive_invalid_steps = 10;
    options.max_consecutive_nonmonotonic_steps = 10;

    options.callbacks.push_back(&pImpl->loggingCallback_);
    options.update_state_every_iteration = true;

    ceres::Solver::Summary summary;
    pImpl->loggingCallback_.bundleAdjusterIteration_++;
    ceres::Solve(options, pImpl->problem_.get(), &summary);
    std::cout << summary.FullReport() << "\n";

    pImpl->loggingCallback_.SaveToFile();
}

void BundleAdjuster::addPointToProblem(double observed_px, double observed_py,
                                       double* camera_matrix, double* point_3d)
{
    ceres::CostFunction* cost_function =
        BundleAdjuster::impl::SnavelyReprojectionError::Create(observed_px,
                                                               observed_py);
    auto id = pImpl->problem_->AddResidualBlock(cost_function, pImpl->loss_function_.get(),
                                                camera_matrix, point_3d,
                                                pImpl->cameraIntrinsics.data());

    pImpl->residual_ids_.push_back(id);
}

size_t BundleAdjuster::addObservation(size_t cameraId, size_t worldPointId,
                                      size_t landmarkId, double px, double py)
{
    BundleAdjuster::Observation observation;
    observation.cameraId = cameraId;
    observation.worldPointId = worldPointId;
    observation.landmarkId = landmarkId;
    observation.px = px;
    observation.py = py;
    pImpl->observations.emplace_back(observation);
    pImpl->residualFileFs_ << observation.cameraId << ","
                           << observation.worldPointId << ","
                           << observation.landmarkId << "," << px << "," << py
                           << "," << px << ","
                           << pImpl->worldPoints[observation.landmarkId * 3]
                           << ","
                           << pImpl->worldPoints[observation.landmarkId * 3 + 1]
                           << ","
                           << pImpl->worldPoints[observation.landmarkId * 3 + 2]
                           << "\n";
    pImpl->residualFileFs_.flush();

    return pImpl->observations.size() - 1;
}

void BundleAdjuster::addCameraParam(double* camera, size_t cameraId)
{
    for (size_t i = 0; i < 6; ++i)
    {
        pImpl->cameras[cameraId * 6 + i] = (camera[i]);
    }
}

void BundleAdjuster::addWorldPointParam(double* worldPoint, size_t landmarkId)
{
    pImpl->worldPoints[landmarkId * 3] = worldPoint[0];
    pImpl->worldPoints[landmarkId * 3 + 1] = worldPoint[1];
    pImpl->worldPoints[landmarkId * 3 + 2] = worldPoint[2];
}

void BundleAdjuster::addCameraIntrinsicsParam(double* cameraIntrinsics)
{
    pImpl->cameraIntrinsics[0] = cameraIntrinsics[0];
    pImpl->cameraIntrinsics[1] = cameraIntrinsics[1];
    pImpl->cameraIntrinsics[2] = cameraIntrinsics[2];
    pImpl->problem_->AddParameterBlock(pImpl->cameraIntrinsics.data(), 3);
    pImpl->problem_->SetParameterBlockConstant(pImpl->cameraIntrinsics.data());
}

double* BundleAdjuster::mutable_camera_for_observation(size_t cameraId)
{
    return &pImpl->cameras[cameraId * 6];
}

double* BundleAdjuster::mutable_point_for_observation(size_t landmarkId)
{
    return &pImpl->worldPoints[landmarkId * 3];
}

const std::vector<BundleAdjuster::Observation>&
BundleAdjuster::getObservations()
{
    return pImpl->observations;
}

const std::vector<double>& BundleAdjuster::getCameras()
{
    return pImpl->cameras;
}

const std::vector<double>& BundleAdjuster::getPoints()
{
    return pImpl->worldPoints;
}

void BundleAdjuster::clearAllResiduals()
{
    for (ceres::ResidualBlockId id : pImpl->residual_ids_)
    {
        pImpl->problem_->RemoveResidualBlock(id);
    }
    pImpl->residual_ids_.clear();
}