#include "AppConfig.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

AppConfig::AppConfig()
{
    //AppConfig("../resources/config/config.json");
    //Do nothing, empty constructor, dangerous, and should probably be fixed. 
    //Still need to figure out how to have config variable accesible everywhere in main.cpp
}

AppConfig::AppConfig(const std::string& filename): 
    m_run_local_bundle_adjustment(false),
    m_run_global_bundle_adjustment(false),
    m_loss_function_type("CAUCHY")
{
    // read filename from config file
    std::ifstream configFile(filename);
    nlohmann::json configJson = nlohmann::json::parse(configFile);

    m_datasetPath = configJson["dataset_path"];
    spdlog::info("Dataset path: {}", m_datasetPath);

    m_runHeadless = configJson["run_headless"];
    spdlog::info("Run headless?: {}", m_runHeadless);

    m_write_ground_truth_camera_poses = configJson["write_ground_truth_camera_poses"];
    spdlog::info("Write ground truth camera poses?: {}", m_write_ground_truth_camera_poses);

    m_ground_truth_poses_scale_type = configJson["ground_truth_poses_scale_type"];
    spdlog::info("Ground truth poses scale type: {}", m_ground_truth_poses_scale_type);

    m_ground_truth_poses_scale_factor = configJson["ground_truth_poses_scale_factor"];
    spdlog::info("Ground truth poses scale factor: {}", m_ground_truth_poses_scale_factor);

    m_feature_match_ratio_threshold = configJson["feature_match_ratio_threshold"];
    spdlog::info("Feature match ratio threshold: {}", m_feature_match_ratio_threshold);

    m_frame_interval = configJson["frame_interval"];
    spdlog::info("Frame interval: {}", m_frame_interval);

    m_dataset_has_depth = configJson["dataset_has_depth"];
    spdlog::info("Dataset has depth?: {}", m_dataset_has_depth);

    m_dataset_has_ground_truth_trajectory = configJson["dataset_has_ground_truth_trajectory"];
    spdlog::info("Dataset has ground truth trajectory?: {}", m_dataset_has_ground_truth_trajectory);

    m_camera_type = configJson["camera_type"];
    spdlog::info("Camera type: {}", m_camera_type);

    m_num_frames = configJson["num_frames"];
    spdlog::info("Number of frames: {}", m_num_frames);

    m_blurry_threshold = configJson["blurry_threshold"];
    spdlog::info("Blurry threshold: {}", m_blurry_threshold);

    num_features_to_detect = configJson["num_features_to_detect"];
    spdlog::info("Number of features to detect: {}", num_features_to_detect);

    min_reprojection_error_threshold = configJson["min_reprojection_error_threshold"];
    spdlog::info("Minimum reprojection error threshold: {}", min_reprojection_error_threshold);

    reprojection_error_max_std_devs = configJson["reprojection_error_max_std_devs"];
    spdlog::info("Reprojection error max standard deviations: {}", reprojection_error_max_std_devs);

    if (configJson.contains("run_local_bundle_adjustment"))
    {
        m_run_local_bundle_adjustment = configJson["run_local_bundle_adjustment"];
    }
    spdlog::info("Run local bundle adjustment?: {}", m_run_local_bundle_adjustment);

    if (configJson.contains("run_global_bundle_adjustment"))
    {
        m_run_global_bundle_adjustment = configJson["run_global_bundle_adjustment"];
    }
    spdlog::info("Run global bundle adjustment?: {}", m_run_global_bundle_adjustment);

    if (configJson.contains("filter_if_not_seen"))
    {
        filter_if_not_seen = configJson["filter_if_not_seen"];
    }
    spdlog::info("Filter if not seen?: {}", filter_if_not_seen);

    if (configJson.contains("filter_if_not_seen_n_times"))
    {
        filter_if_not_seen_n_times = configJson["filter_if_not_seen_n_times"];
    }
    spdlog::info("Filter if not seen n times?: {}", filter_if_not_seen_n_times);

    if (configJson.contains("filter_if_not_seen_n_times_in_m_frames"))
    {
        filter_if_not_seen_n_times_in_m_frames = configJson["filter_if_not_seen_n_times_in_m_frames"];
    }
    spdlog::info("Filter if not seen n times in m frames?: {}", filter_if_not_seen_n_times_in_m_frames);

    if (configJson.contains("loss_function_type"))
    {
        m_loss_function_type = configJson["loss_function_type"];
    }
}

//AppConfig AppConfig::create(const std::string& filename)
//{
//    // read filename from config file
//    std::ifstream configFile(filename);
//    nlohmann::json configJson = nlohmann::json::parse(configFile);
//
//    m_datasetPath = configJson["dataset_path"];
//    spdlog::info("Dataset path: {}", m_datasetPath);
//
//    m_runHeadless = configJson["run_headless"];
//    spdlog::info("Run headless?: {}", m_runHeadless);
//}

std::string AppConfig::getDatasetPath() const
{
    return m_datasetPath;
}

bool AppConfig::getRunHeadless() const
{
    return m_runHeadless;
}

bool AppConfig::getWriteGroundTruthCameraPoses() const
{
    return m_write_ground_truth_camera_poses;
}

int AppConfig::getGroundTruthPosesScaleType() const
{
    return m_ground_truth_poses_scale_type;
}

float AppConfig::getGroundTruthPosesScaleFactor() const
{
    return m_ground_truth_poses_scale_factor;
}

float AppConfig::getFeatureMatchRatioThreshold() const
{
    return m_feature_match_ratio_threshold;
}

size_t AppConfig::getFrameInterval() const
{
    return m_frame_interval;
}

bool AppConfig::getDatasetHasDepth() const
{
    return m_dataset_has_depth;
}

bool AppConfig::getDatasetHasGroundTruthTrajectory() const
{
    return m_dataset_has_ground_truth_trajectory;
}

std::string AppConfig::getCameraType() const
{
    return m_camera_type;
}

size_t AppConfig::getNumFrames() const
{
    return m_num_frames;
}

float AppConfig::getBlurryThreshold() const
{
    return m_blurry_threshold;
}

int AppConfig::getNumFeaturesToDetect() const
{
    return num_features_to_detect;
}

float AppConfig::getMinReprojectionErrorThreshold() const
{
    return min_reprojection_error_threshold;
}

float AppConfig::getReprojectionErrorMaxStdDevs() const
{
    return reprojection_error_max_std_devs;
}

bool AppConfig::getRunLocalBundleAdjustment() const
{
    return m_run_local_bundle_adjustment;
}

bool AppConfig::getRunGlobalBundleAdjustment() const
{
    return m_run_global_bundle_adjustment;
}

bool AppConfig::getFilterIfNotSeen() const
{
    return filter_if_not_seen;
}

size_t AppConfig::getFilterIfNotSeenNTimes() const
{
    return filter_if_not_seen_n_times;
}

size_t AppConfig::getFilterIfNotSeenNTimesInMFrames() const
{
    return filter_if_not_seen_n_times_in_m_frames;
}

std::string AppConfig::getLossFunctionType() const
{
    return m_loss_function_type;
}