#ifndef BUNDLE_ADJUSTER_H
#define BUNDLE_ADJUSTER_H

#include <memory>
#include <vector>
#include <string>

class BundleAdjuster
{
    struct Observation
    {
        size_t cameraId;
        size_t worldPointId;
        size_t landmarkId;
        double px;
        double py;
    };

    enum LossFunctionType
    {
        TRIVIAL,
        HUBER,
        CAUCHY
    };

   public:
    BundleAdjuster(std::string logFilePath, const std::string& lossFunctionType = "CAUCHY");
    ~BundleAdjuster();
    size_t addObservation(size_t cameraId, size_t worldPointId,
                          size_t landmarkId, double px, double py);

    double* mutable_camera_for_observation(size_t i);
    double* mutable_point_for_observation(size_t i);

    // store optimization problem parameters
    // Ceres requires these parameters in double format (as default precision)
    // additionally, Ceres requires reference to data
    void addCameraParam(double* camera, size_t cameraId);
    void addWorldPointParam(double* worldPoin, size_t landmarkId);
    void addCameraIntrinsicsParam(
        double* camera);  // camera intrinsics added as constant parameter block
                          // (not optimized for)

    // setup Ceres problem
    void setupProblem();

    // add residual block
    void addPointToProblem(double observed_px, double observed_py,
                           double* camera_matrix, double* point_3d);

    // solve ceres problem
    void solve();

    // get const reference to output parameter data so we can use optimized
    // values
    const std::vector<BundleAdjuster::Observation>& getObservations();
    const std::vector<double>& getCameras();
    const std::vector<double>& getPoints();

    // clear residual blocks from problem -- allows us to reuse Ceres problem
    void clearAllResiduals();

   private:
    // PIMPL idiom: enables faster compilation, hides implementaiton details
    struct impl;
    std::unique_ptr<impl> pImpl;
};

#endif  // BUNDLE_ADJUSTER_H