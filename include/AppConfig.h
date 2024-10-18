#ifndef APP_CONFIG
#define APP_CONFIG

#include <string>

class AppConfig
{
public:
    AppConfig();
    AppConfig(const std::string& filename);
    std::string getDatasetPath() const;
    std::string getLossFunctionType() const;
    bool getRunHeadless() const;
    bool getWriteGroundTruthCameraPoses() const;
    int getGroundTruthPosesScaleType() const;
    float getGroundTruthPosesScaleFactor() const;
    float getFeatureMatchRatioThreshold() const;
    size_t getFrameInterval() const;
    bool getDatasetHasDepth() const;
    bool getDatasetHasGroundTruthTrajectory() const;
    std::string getCameraType() const;
    size_t getNumFrames() const;
    float getBlurryThreshold() const;
    int getNumFeaturesToDetect() const;
    float getMinReprojectionErrorThreshold() const;
    float getReprojectionErrorMaxStdDevs() const;
    bool getRunLocalBundleAdjustment() const;
    bool getRunGlobalBundleAdjustment() const;
    bool getFilterIfNotSeen() const;
    size_t getFilterIfNotSeenNTimes() const;
    size_t getFilterIfNotSeenNTimesInMFrames() const;

private:
    std::string m_datasetPath;
    bool m_runHeadless;
    bool m_write_ground_truth_camera_poses;
    int m_ground_truth_poses_scale_type;
    float m_ground_truth_poses_scale_factor;
    float m_feature_match_ratio_threshold;
    size_t m_frame_interval;
    bool m_dataset_has_depth;
    bool m_dataset_has_ground_truth_trajectory;
    std::string m_camera_type;
    size_t m_num_frames;
    float m_blurry_threshold;
    int num_features_to_detect;    
    float min_reprojection_error_threshold;
    float reprojection_error_max_std_devs;
    bool m_run_global_bundle_adjustment;
    bool m_run_local_bundle_adjustment;
    bool filter_if_not_seen;
    size_t filter_if_not_seen_n_times;
    size_t filter_if_not_seen_n_times_in_m_frames;
    std::string m_loss_function_type;
};

#endif // APP_CONFIG