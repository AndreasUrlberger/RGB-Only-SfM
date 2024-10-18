#include "Eigen.h"
#include <opencv2/core/cvdef.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <array>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/quaternion.hpp>
#include <tuple>
#include <vector>

// JSON config
#include "AppConfig.h"

// logging
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"

#include "VirtualSensor.h"

#include "ceres/ceres.h"
#include "SimpleMesh.h"

#include "BundleAdjuster.h"
#include "Utils.h"

const size_t DEBUG_LANDMARK = 2;

const bool detailedPrint = true;

std::string matToString(const cv::Mat& mat) 
{
    std::ostringstream oss;
    oss << "Matrix size: " << mat.rows << "x" << mat.cols << "\n";
    oss << "Matrix type: " << mat.type() << "\n";
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            oss << mat.at<float>(i, j) << " ";
        }
        oss << "\n";
    }
    return oss.str();
}

class StructureFromMotion {
public:
		//Taken from ............., helps keep track of all frames and detected features/matches
	struct SFM_Helper
	{
		struct ProcessedFrame
		{
            cv::Mat cameraPose;
            cv::Mat projectionMatrix;
            cv::Mat featureDescriptor;
            cv::Mat image;
            std::vector<cv::KeyPoint> keypoints;
            std::vector<cv::Point2f> keypointsFloat;

		};

		std::vector<ProcessedFrame> processed_frames;
	};

	struct State_Data 
	{
		VirtualSensor sensor;
		cv::Mat colorIntrinsics;
        cv::Mat depthIntrinsics;
		AppConfig config;
        float groundTruthScaleFactor;

        int frameIndex = -1;

        // cv::Ptr<cv::AKAZE> detector; @TODO remove
        cv::Ptr<cv::SIFT> detector;

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> matches;

        std::vector<cv::Point3f> landmarks;
        std::vector<size_t> numSeen;
        // Stores index of feature descriptor corresponding to each world point. (index of world point, index of feature descriptor) -> (index of source frame, index of feature descriptor)
        std::vector<std::vector<std::tuple<size_t, size_t>>> featureDescriptorIndices;
        std::vector<Vector4uc> pointColors;

        std::vector<cv::Mat> groundTruthPoses;
	};

    ~StructureFromMotion()
    {
        reprojectionErrorFs.close();
        testMetricsOutputFS.close();
        if (bundleAdjuster) {
            delete bundleAdjuster;
            bundleAdjuster = nullptr;
        }
    }


	State_Data data;
	SFM_Helper SFM;
    size_t goodPointsIdx = 0;
    BundleAdjuster *bundleAdjuster;

	// Memory allocation for image stuff
	cv::Mat DebugOutputImage;

	int DEBUG_SFM = 1;
	int N_DETECT_FEATURES;
    int numBlurryFramesSkipped = 0;
    bool runHeadless;
    std::string datasetPath;
    std::string datasetName;

    bool writeGroundTruth;
    int scaleFactorType;
    float config_groundTruthScaleFactor;
    float match_ratio_thresh;
    ImageType typeOfImage;
    bool hasDepth;
    bool hasTrajectory;
    float blurThreshold;
    int process_frames;
    int frameInterval;
    float min_reprojection_error_threshold;
    float reprojection_error_max_std_devs;
    std::ofstream reprojectionErrorFs;
    bool filter_if_not_seen;
    size_t filter_if_not_seen_n_times;
    size_t filter_if_not_seen_n_times_in_m_frames;

    std::ofstream testMetricsOutputFS;

    void getObserved(
        const std::vector<cv::Point3f>& landmarks,
        const std::vector<std::vector<std::tuple<size_t, size_t>>>& featureDescriptorIndices,
        const std::vector<SFM_Helper::ProcessedFrame>& processed_frames,
        std::vector<cv::Point2f>& observed
    )
    {
        for (size_t j = 0; j < landmarks.size(); ++j)
        {
            for (size_t k = 0;
                    k < featureDescriptorIndices[j].size(); ++k)
            {
                size_t frameId =
                    std::get<0>(featureDescriptorIndices[j][k]);
                size_t descId =
                    std::get<1>(featureDescriptorIndices[j][k]);
                
                observed.push_back(SFM.processed_frames[frameId].keypoints[descId].pt);
            }
        }
        std::cout << "observed 1: " << observed[0] << std::endl;
    }

    void getPredicted(
        const std::vector<cv::Point3f>& landmarks,
        const std::vector<std::vector<std::tuple<size_t, size_t>>>& featureDescriptorIndices,
        const std::vector<SFM_Helper::ProcessedFrame>& processed_frames,
        const cv::Mat& colorIntrinsics,
        std::vector<cv::Point2f>& predicted
    )
    {
        //Produces the projected 2D points

        //TODO: try to replace some or all of the below with the OpenCV ProjectPoints Method



        for (size_t j = 0; j < landmarks.size(); ++j)
        {
            for (size_t k = 0; k < featureDescriptorIndices[j].size(); ++k)
            {
                size_t frameId =
                    std::get<0>(featureDescriptorIndices[j][k]);
                size_t descId =
                    std::get<1>(featureDescriptorIndices[j][k]);

                cv::Mat rot = GetInverseTransformation(
                    processed_frames[frameId].cameraPose)(
                    cv::Range(0, 3), cv::Range(0, 3));
                cv::Mat rvec;
                cv::Rodrigues(
                    GetInverseTransformation(
                        processed_frames[frameId].cameraPose)(
                        cv::Rect(0, 0, 3, 3)),
                    rvec);

                auto p = landmarks[j];
                // std::cout << "world point before project: " << p <<
                // std::endl; std::cout << "pose: " <<
                // GetInverseTransformation(SFM.processed_frames[frameId].cameraPose)
                // << std::endl; std::cout << "instrinsics matrix: " <<
                // data.colorIntrinsics << "\n";

                cv::Mat cameraPoint =
                    GetInverseTransformation(
                        processed_frames[frameId].cameraPose) *
                    (cv::Mat_<float>(4, 1) << p.x, p.y, p.z, 1);
                // std::cout << "camera point: " << cameraPoint <<
                // std::endl;

                //cv::Mat projected;
                //cv::projectPoints(landmarks, rvec, p, colorIntrinsics, cv::noArray(), projected);

                cv::Mat projected =
                    colorIntrinsics *
                    cameraPoint(cv::Range(0, 3), cv::Range(0, 1));
                projected.at<float>(0) =
                    projected.at<float>(0) / projected.at<float>(2);
                projected.at<float>(1) =
                    projected.at<float>(1) / projected.at<float>(2);
                    
                predicted.push_back({ projected.at<float>(0), projected.at<float>(1) });
            }
        }
    }
	/**
	 * Initialisation, set variables based on config file, create sensor object, read files,
	 */
	
    bool Init() 
	{
		// Set log level
		spdlog::cfg::load_env_levels();

		// Read JSON config
		std::string configPath = "../resources/config/config.json";
		std::string envConfigPath = getenv("CONFIG_PATH") ? getenv("CONFIG_PATH") : "";
		if (!envConfigPath.empty())
		{
			configPath = envConfigPath;
		}

		spdlog::info("Reading from config file: {}", configPath);
        if (!LoadConfig(configPath)){
            std::cerr << "Failed to load config file arguments\n";
            return false;
        }

		std::string filenameBaseOut = "mesh_";

		// load video
		std::cout << "Initialize virtual sensor..." << std::endl;
		if (!data.sensor.Init(datasetPath, typeOfImage, hasDepth, hasTrajectory))
		{
				std::cout
					<< "Failed to initialize the sensor!\nCheck file path!"
					<< std::endl;
				return false;
		}

		// Define camera intrinsics, needed as type of cv::Mat
		data.colorIntrinsics = data.sensor.GetColorIntrinsics(1);

        // Create output directory in build folder
        datasetName = datasetPath;
        // Remove trailing slash
        if (!datasetName.empty() && (datasetName.back() == '/' || datasetName.back() == '\\')) {
            datasetName.pop_back();
        }
        // Take only the last part of the path
        datasetName = datasetName.substr(datasetName.find_last_of("/\\") + 1);
        std::filesystem::create_directory(datasetName);

        bundleAdjuster = new BundleAdjuster(datasetName + "/log_loss_" + GetConfigOptionsString() + ".csv", data.config.getLossFunctionType());
        bundleAdjuster->setupProblem();
        reprojectionErrorFs.open(datasetName + "/reprojection_error_" + GetConfigOptionsString() + ".csv", std::ios::trunc);

        testMetricsOutputFS.open(datasetName + "/Metrics_Output_" + GetConfigOptionsString() + ".csv", std::ios::trunc);
        //testMetricsOutputFS << "frame,time,total_time,n_world_points,total_repr_err,avg_rep_err_(per_point)\n";
        testMetricsOutputFS << "frame,time,total_time,n_world_points,rms_repr_err,total_repr_err,avg_rep_err_(per_point),cam_mean_t_x_err,cam_mean_t_y_err,cam_mean_t_z_err,cam_std_t_x_err,cam_std_t_y_err,cam_std_t_z_err,cam_min_t_x_err,cam_min_t_y_err,cam_min_t_z_err,cam_max_t_x_err,cam_max_t_y_err,cam_max_t_z_err,cam_mean_r_x_err,cam_mean_r_y_err,cam_mean_r_z_err,cam_std_r_x_err,cam_std_r_y_err,cam_std_r_z_err,cam_min_r_x_err,cam_min_r_y_err,cam_min_r_z_err,cam_max_r_x_err,cam_max_r_y_err,cam_max_r_z_err \n";



        //Setup the feature detector and matcher
		if (N_DETECT_FEATURES == -1) {
			data.detector = cv::SiftFeatureDetector::create();		// Can also use ORB
		}
		else 
		{
			data.detector = cv::SiftFeatureDetector::create(N_DETECT_FEATURES);		// Can also use ORB
		}
		
		data.matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        // Actual frame increment depends on blurryness of the images. Do not change.
		data.sensor.SetIncrement(1);

		return true;
	}

    bool LoadConfig(std::string configPath)
    {
        data.config = AppConfig(configPath);
        writeGroundTruth = data.config.getWriteGroundTruthCameraPoses();
        scaleFactorType = data.config.getGroundTruthPosesScaleType();
        config_groundTruthScaleFactor = data.config.getGroundTruthPosesScaleFactor();
        match_ratio_thresh = data.config.getFeatureMatchRatioThreshold();
        process_frames = data.config.getNumFrames();
        frameInterval = data.config.getFrameInterval();
        hasDepth = data.config.getDatasetHasDepth();
        hasTrajectory = data.config.getDatasetHasGroundTruthTrajectory();
        blurThreshold = data.config.getBlurryThreshold();
        datasetPath = data.config.getDatasetPath();
        runHeadless = data.config.getRunHeadless();
        N_DETECT_FEATURES = data.config.getNumFeaturesToDetect();
        min_reprojection_error_threshold = data.config.getMinReprojectionErrorThreshold();
        reprojection_error_max_std_devs = data.config.getReprojectionErrorMaxStdDevs();
        filter_if_not_seen = data.config.getFilterIfNotSeen();
        filter_if_not_seen_n_times = data.config.getFilterIfNotSeenNTimes();
        filter_if_not_seen_n_times_in_m_frames = data.config.getFilterIfNotSeenNTimesInMFrames();
        if (data.config.getCameraType() == "kinect")
        {
            typeOfImage = ImageType::Kinect;
        }
        else if (data.config.getCameraType() == "colmap")
        {
            typeOfImage = ImageType::Colmap;
        }
        else if (data.config.getCameraType() == "colmap_640_480")
        {
            typeOfImage = ImageType::Colmap_640_480;
        }
        else
        {
            std::cerr << "Unknown camera type in config file\n";
            return false;
        }

        return true;
    }

    bool isBlurryImage(cv::Mat &image, float &variance)
    {
        int kernelSize = 9;
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(kernelSize, kernelSize), 0);
        gray.convertTo(gray, CV_64F);
        blurred.convertTo(blurred, CV_64F);
        cv::Mat diff = blurred - gray;
        cv::Scalar mean, stddev;

        cv::meanStdDev(diff, mean, stddev);
        variance = stddev.val[0] * stddev.val[0] / (kernelSize * kernelSize);

        if (detailedPrint) std::cout << "Variance diff: " << variance  << "\n";

        return variance < blurThreshold;
    }

    bool isBlurryImage(cv::Mat &image)
    {
        float variance;
        return isBlurryImage(image, variance);
    }

    bool LoadNextImage()
    {
        bool endOfVideo = false;
        cv::Mat lastGoodFrame;
        bool hasGoodFrame = false;
        int numIterations = 0;

        int framesToSkip;

        if (data.frameIndex == 0){
            // First frame, don't skip x frames.
            framesToSkip = 0;
        }
        else
        {
            // Skip x frames after each frame.
            framesToSkip = frameInterval - 1;
        }

        // Skip x frames but keep latest good image as backup in case we cannot find any better in the x frames.
        do 
        {
            endOfVideo = !data.sensor.ProcessNextFrame();

            if (!endOfVideo)
            {
                // Supposedly, the cv::Mat is using a smart pointer to the data, so we have to clone the Mat wrapper to get a deep copy of the data but can then safely just add the Mat to the vector.
                cv::Mat frame = cv::Mat(data.sensor.GetColorImageHeight(), data.sensor.GetColorImageWidth(), CV_8UC4, data.sensor.GetColorRGBX()).clone();
                const bool blurry = isBlurryImage(frame);

                if (!blurry)
                {
                    lastGoodFrame = frame;
                    hasGoodFrame = true;
                }
            }

            --framesToSkip;
        }
        while((!endOfVideo) && (framesToSkip > 0));

        // Continue searching for good frame if we have not found any yet.
        while((!endOfVideo) && (!hasGoodFrame))
        {
            endOfVideo = !data.sensor.ProcessNextFrame();

            if (!endOfVideo)
            {
                cv::Mat frame = cv::Mat(data.sensor.GetColorImageHeight(), data.sensor.GetColorImageWidth(), CV_8UC4, data.sensor.GetColorRGBX()).clone();
                const bool blurry = isBlurryImage(frame);

                if (!blurry)
                {
                    lastGoodFrame = frame;
                    hasGoodFrame = true;
                }
            }
        }

        // Check if we found a good frame.
        if (hasGoodFrame)
        {
            SFM.processed_frames.push_back(SFM_Helper::ProcessedFrame());
            SFM_Helper::ProcessedFrame &processedFrame = SFM.processed_frames.back();
            processedFrame.image = lastGoodFrame;
            ++data.frameIndex;

            if (writeGroundTruth)
            {
                Eigen::Matrix4f groundTruthPoseEigen = data.sensor.GetTrajectory();
                cv::Mat groundTruthPose = cv::Mat::eye(4, 4, CV_64F);
                cv::eigen2cv(groundTruthPoseEigen, groundTruthPose);
                data.groundTruthPoses.push_back(groundTruthPose);
            }
        }
        else 
        {
            // Must have reached end of video.
            std::cout << "Been through all frames" << "\n";
        }

        return hasGoodFrame;
    }

    void ExtractFeatures(int frameIndex)
    {
        SFM_Helper::ProcessedFrame &processedFrame = SFM.processed_frames[frameIndex];
        data.detector->detectAndCompute(processedFrame.image, cv::noArray(), processedFrame.keypoints, processedFrame.featureDescriptor);
    }

    cv::Mat ComposeTransformationMatrix(cv::Mat &rotation, cv::Mat &translation)
    {
        cv::Mat transformation = cv::Mat::eye(4, 4, CV_32F);
        rotation.copyTo(transformation(cv::Rect(0, 0, 3, 3)));
        translation.copyTo(transformation(cv::Rect(3, 0, 1, 3)));
        return transformation;
    }

    void FindGoodFeatureMatches(int frameIndex1, int frameIndex2, std::vector<cv::Point2f> &floatKeypoints1, std::vector<cv::Point2f> &floatKeypoints2, std::vector<cv::DMatch> &matches)
    {
        SFM_Helper::ProcessedFrame &processedFrame1 = SFM.processed_frames[frameIndex1];
        SFM_Helper::ProcessedFrame &processedFrame2 = SFM.processed_frames[frameIndex2];
        std::vector<std::vector<cv::DMatch>> allMatches;

        data.matcher->knnMatch(processedFrame1.featureDescriptor, processedFrame2.featureDescriptor, allMatches, 2);

        //Filter allMatches
        for (size_t i = 0; i < allMatches.size(); ++i)
        {
            if (allMatches[i][0].distance < match_ratio_thresh * allMatches[i][1].distance)
            {
                matches.push_back(allMatches[i][0]);
            }
        }

        if (detailedPrint) std::cout << "Number of good feature matches " << matches.size() << "\n";

        // Display the found matches
        cv::drawMatches(processedFrame1.image, processedFrame1.keypoints, processedFrame2.image, processedFrame2.keypoints, matches, DebugOutputImage);
        DebugDisplay(frameIndex1);

        // Convert to float points
        std::vector<cv::Point2f> tempPoints1, tempPoints2;
        cv::KeyPoint::convert(processedFrame1.keypoints, tempPoints1);
        cv::KeyPoint::convert(processedFrame2.keypoints, tempPoints2);

        for (size_t index = 0; index < matches.size(); ++index)
        {
            floatKeypoints1.push_back(tempPoints1[matches[index].queryIdx]);
            floatKeypoints2.push_back(tempPoints2[matches[index].trainIdx]);
        }
    }

    void mapFeaturesToDepth(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& depthImage,
        std::vector<Eigen::Vector3d>& points3D) 
    {
        for (const auto& kp : keypoints) 
        {
            float depth = depthImage.at<uint16_t>(kp.pt.y, kp.pt.x); // maybe add * 0.001f
            
            if (depth > 0) 
            {
                Eigen::Vector3d point;
                cv::Mat intrinsics = data.depthIntrinsics;  //Apply intrinsic matrix to depth image, Not sure if for TUM this is already done
                point[0] = (kp.pt.x - intrinsics.at<float>(0, 2)) * depth / intrinsics.at<float>(0, 0);
                point[1] = (kp.pt.y - intrinsics.at<float>(1, 2)) * depth / intrinsics.at<float>(1, 1);
                point[2] = depth;

                points3D.push_back(point);
            }
        }
    }

    // Returns the rotation and translation from the first camera to the second camera, i.e. the rotation and translation of the second camera in the coordinate system of the first camera.
    cv::Mat EstimateUnscaledTransformation(int pointIndex1, int pointIndex2, cv::Mat &rotation, cv::Mat &translation)
    {
        std::vector<cv::Point2f> points1 = SFM.processed_frames[pointIndex1].keypointsFloat;
        std::vector<cv::Point2f> points2 = SFM.processed_frames[pointIndex2].keypointsFloat;

        if (points1.size() < 5)
        {
            std::cout << "Not enough matches found to compute essential matrix, need at least 5 matches" << "\n";
            return cv::Mat();
        }

        cv::Mat inliersMask;
        // Compute the essential matrix from the corresponding points in the two images, i.e. the matrix that fulfills the epipolar constraint. (Uses the 5-point algorithm).
        cv::Mat essentialMatrix = cv::findEssentialMat(points1, points2, data.colorIntrinsics, cv::RANSAC, 0.999, 1.0, inliersMask);

        // Decomposes the essential matrix and returns the pair of rotation and translation for which all points are in front of both cameras (chirality check).
        // Important: Translation is always scaled to have a norm of 1. We do not get any information about the scale of the translation.
        int inliers = cv::recoverPose(essentialMatrix, points1, points2, data.colorIntrinsics, rotation, translation, inliersMask); // Basis transformation for points from camera 1 to camera 2. inv(relativePose1to2) = relative extrinsics cam 1 to cam 2.

        // Could also just pass the points in the other order to get the rotation and translation directly.
        rotation = rotation.t();
        translation = -rotation * translation;

        return inliersMask;
    }

    cv::Mat GetInverseTransformation(cv::Mat const &transformation)
    {
        cv::Mat inverseTransformation = cv::Mat::eye(4, 4, CV_32F);
        cv::Mat rotation = transformation(cv::Rect(0, 0, 3, 3));
        cv::Mat translation = transformation(cv::Rect(3, 0, 1, 3));
        cv::Mat inverseRotation = rotation.t();
        cv::Mat inverseTranslation = -inverseRotation * translation;
        inverseRotation.copyTo(inverseTransformation(cv::Rect(0, 0, 3, 3)));
        inverseTranslation.copyTo(inverseTransformation(cv::Rect(3, 0, 1, 3)));
        return inverseTransformation;
    }

    size_t RemoveReprojectionOutliers(std::vector<cv::Point3f> &landmarks, cv::Mat &inliersMask, cv::Mat &extrinsics1, cv::Mat &extrinsics2, std::vector<cv::Point2f> &featurePoints1, std::vector<cv::Point2f> &featurePoints2)
    {
        cv::Mat reprojectedPoints1, reprojectedPoints2, rvec1, rvec2;
        cv::Rodrigues(extrinsics1(cv::Rect(0, 0, 3, 3)), rvec1);
        cv::Rodrigues(extrinsics2(cv::Rect(0, 0, 3, 3)), rvec2);
        cv::projectPoints(landmarks, rvec1, extrinsics1(cv::Rect(3, 0, 1, 3)), data.colorIntrinsics, cv::noArray(), reprojectedPoints1);
        cv::projectPoints(landmarks, rvec2, extrinsics2(cv::Rect(3, 0, 1, 3)), data.colorIntrinsics, cv::noArray(), reprojectedPoints2);

        std::vector<float> perPointReproError1, perPointReproError2;
        float totalReproError1 = 0.0f;
        float totalReproError2 = 0.0f;
        size_t numInliers = 0;
        for (size_t index = 0; index < landmarks.size(); ++index)
        {
            if (inliersMask.at<uchar>(index) == 1)
            {
                // Inlier
                ++numInliers;
                const float reproError1 = cv::norm(featurePoints1[index] - reprojectedPoints1.at<cv::Point2f>(index));
                const float reproError2 = cv::norm(featurePoints2[index] - reprojectedPoints2.at<cv::Point2f>(index));
                perPointReproError1.push_back(reproError1);
                perPointReproError2.push_back(reproError2);
                totalReproError1 += reproError1;
                totalReproError2 += reproError2;
            }
            else {
                // Outlier
                // Just add them to keep indexing easier.
                perPointReproError1.push_back(0.0f);
                perPointReproError2.push_back(0.0f);
            }
        }

        const float meanReproError1 = totalReproError1 / numInliers;
        const float meanReproError2 = totalReproError2 / numInliers;
        float variance1 = 0.0f;
        float variance2 = 0.0f;
        for (size_t index = 0; index < landmarks.size(); ++index)
        {
            if (inliersMask.at<uchar>(index) == 1)
            {
                variance1 += std::pow(perPointReproError1[index] - meanReproError1, 2);
                variance2 += std::pow(perPointReproError2[index] - meanReproError2, 2);
            }
        }

        const float standardDeviation1 = std::sqrt(variance1 / numInliers);
        const float standardDeviation2 = std::sqrt(variance2 / numInliers);
        // We keep a minimum threshold of 2.0 pixels to not remove too many and too good points.
        const float threshold1 = fmax(meanReproError1 + reprojection_error_max_std_devs * standardDeviation1, min_reprojection_error_threshold);
        const float threshold2 = fmax(meanReproError2 + reprojection_error_max_std_devs * standardDeviation2, min_reprojection_error_threshold);

        if (detailedPrint) std::cout << "Mean reprojection error 1: " << meanReproError1 << ", standard deviation 1: " << standardDeviation1 << ", threshold 1: " << threshold1 << "\n";
        if (detailedPrint) std::cout << "Mean reprojection error 2: " << meanReproError2 << ", standard deviation 2: " << standardDeviation2 << ", threshold 2: " << threshold2 << "\n";

        size_t numOutliersRemoved = 0;
        for (int index = landmarks.size() - 1; index >= 0; --index)
        {
            if (inliersMask.at<uchar>(index) == 1)
            {
                if ((perPointReproError1[index] > threshold1) || (perPointReproError2[index] > threshold2))
                {
                    inliersMask.at<uchar>(index) = 0;
                    ++numOutliersRemoved;
                }
            }
        }

        return numOutliersRemoved;
    }

	bool InitialPoseEstimate()
	{
        auto t1 = std::chrono::high_resolution_clock::now();
        // Load the first two frames
        if (!LoadNextImage() || !LoadNextImage())
        {
            std::cout << "Not enough frames to estimate initial pose" << "\n";
            return false;
        }
        SFM_Helper::ProcessedFrame &processedFrame1 = SFM.processed_frames[0];
        SFM_Helper::ProcessedFrame &processedFrame2 = SFM.processed_frames[1];
        ExtractFeatures(0);
        ExtractFeatures(1);

        std::vector<cv::DMatch> matches;
        FindGoodFeatureMatches(0, 1, processedFrame1.keypointsFloat, processedFrame2.keypointsFloat, matches);

        cv::Mat rotation, translation; // Translation and rotation from camera 1 to 2, i.e. rotation and translation of cam 2 in the coordinate system of cam 1.
        cv::Mat inliersMask = EstimateUnscaledTransformation(0, 1, rotation, translation);
        if (detailedPrint) std::cout << "Number of inliers: " << cv::countNonZero(inliersMask) << "\n";
        // First camera pose is the world origin.
        processedFrame1.cameraPose = cv::Mat::eye(4, 4, CV_32F);
        processedFrame2.cameraPose = ComposeTransformationMatrix(rotation, translation);

        // Triangulate the points.
        processedFrame1.projectionMatrix = data.colorIntrinsics * cv::Mat::eye(3, 4, CV_32F);
        processedFrame2.projectionMatrix = data.colorIntrinsics * cv::Mat::eye(3, 4, CV_32F) * GetInverseTransformation(processedFrame2.cameraPose);
        cv::Mat homogeneousPoints;
        cv::triangulatePoints(processedFrame1.projectionMatrix, processedFrame2.projectionMatrix, processedFrame1.keypointsFloat, processedFrame2.keypointsFloat, homogeneousPoints);
        if (detailedPrint) std::cout << "homogeneous points size: " << homogeneousPoints.size() << "\n";
        std::vector<cv::Point3f> landmarks;
        cv::convertPointsFromHomogeneous(homogeneousPoints.t(), landmarks);

        cv::Mat extrinsics1 = GetInverseTransformation(processedFrame1.cameraPose);
        cv::Mat extrinsics2 = GetInverseTransformation(processedFrame2.cameraPose);
        size_t numOutliersRemoved = RemoveReprojectionOutliers(landmarks, inliersMask, extrinsics1, extrinsics2, processedFrame1.keypointsFloat, processedFrame2.keypointsFloat);
        if (detailedPrint) std::cout << "Number of outliers removed: " << numOutliersRemoved << "\n";

        for (size_t matchIndex = 0; matchIndex < matches.size(); ++matchIndex)
        {
            // Don't add outliers to the landmarks.
            if (inliersMask.at<uchar>(matchIndex) == 0)
            {
                continue;
            }

            const cv::DMatch &match = matches[matchIndex];
            data.landmarks.push_back(landmarks[matchIndex]);
            data.featureDescriptorIndices.push_back(std::vector<std::tuple<size_t, size_t>>());
            data.featureDescriptorIndices.back().push_back(std::make_tuple(0, match.queryIdx));
            data.featureDescriptorIndices.back().push_back(std::make_tuple(1, match.trainIdx));
            data.numSeen.push_back(2);

            if (detailedPrint && match.queryIdx == 53)
            {
                std::cout << "Query index #53: " << std::endl;
                std::cout << "keypoint: " << processedFrame1.keypointsFloat[match.queryIdx] << std::endl;
                std::cout << "landmark: " << landmarks[matchIndex] << std::endl;
            }

            // Add color of landmark
            const cv::Point2f keypoint = processedFrame1.keypointsFloat[matchIndex];
            const cv::Mat &image = processedFrame1.image;
            const int x = keypoint.x;
            const int y = keypoint.y;
            const unsigned char blue = image.at<cv::Vec4b>(y, x)[0];
            const unsigned char green = image.at<cv::Vec4b>(y, x)[1];
            const unsigned char red = image.at<cv::Vec4b>(y, x)[2];
            Vector4uc color = Vector4uc(red, green, blue, 255);
            data.pointColors.push_back(color);
        }

        // PrintReprojectionError(processedFrame1, processedFrame2, 0);

        if (detailedPrint) std::cout << "world points size: " << data.landmarks.size() << "\n";

        auto t2 = std::chrono::high_resolution_clock::now();

        /* Getting number of milliseconds as a double. */
        std::chrono::duration<double, std::milli> timeTaken = t2 - t1;

        if (writeGroundTruth)
        {
            PrintAfterFrameOutput(timeTaken.count());
        }
        return true;
	}

    void FindWorldFrameMatches(std::vector<cv::DMatch> &matches)
    {
        SFM_Helper::ProcessedFrame &currFrame = SFM.processed_frames[data.frameIndex];

        // Find matches between feature points of all world points and the current image.
        std::vector<cv::Mat> worldDescriptors;
        for (size_t pointIndex = 0; pointIndex < data.landmarks.size(); ++pointIndex)
        {
            const size_t numFeatureDescriptors = data.featureDescriptorIndices[pointIndex].size();
            // TODO: Not sure if taking the most recent feature descriptor is the smartest way of doing this.
            // We take the most recent feature descriptor of the world point.
            const std::tuple<size_t, size_t> &descriptorIndex = data.featureDescriptorIndices[pointIndex][numFeatureDescriptors - 1];
            // const std::tuple<size_t, size_t> &descriptorIndex = data.featureDescriptorIndices[pointIndex][0];
            const size_t frameIndex = std::get<0>(descriptorIndex);
            const size_t featureIndex = std::get<1>(descriptorIndex);
            worldDescriptors.push_back(SFM.processed_frames[frameIndex].featureDescriptor.row(featureIndex));
        }

        // Not sure if we can avoid this conversion to Mat.
        cv::Mat worldDescriptorsMat;
        cv::vconcat(worldDescriptors, worldDescriptorsMat);

        std::vector<std::vector<cv::DMatch>> allMatches;
        data.matcher->knnMatch(worldDescriptorsMat, currFrame.featureDescriptor, allMatches, 2);
        if (detailedPrint) std::cout << "Number of all matches " << allMatches.size() << "\n";

        //Filter allMatches
        for (size_t i = 0; i < allMatches.size(); ++i)
        {
            if (allMatches[i][0].distance < match_ratio_thresh * allMatches[i][1].distance)
            {
                matches.push_back(allMatches[i][0]);
            }
        }
        if (detailedPrint) std::cout << "Number of good feature matches " << matches.size() << "\n";
    }

    void InitialPoseEsimateMainLoop()
    {
        // only need to do this once, since camera intrinsics constant currently
        if (data.config.getRunLocalBundleAdjustment() ||
            data.config.getRunGlobalBundleAdjustment())
        {
            packCameraIntrinsics(*bundleAdjuster, data.colorIntrinsics);
        }
        
        for (int i = 2; i < process_frames; ++i)
        {
            auto t1 = std::chrono::high_resolution_clock::now();
            // Remove old frames
            SFM.processed_frames[i - 2].image = cv::Mat();

            if (detailedPrint) std::cout << "Processing frame " << i << "\n";
            // Load next iamge and extract features as well as matches between this and the previous image.
            if (!LoadNextImage())
            {
                std::cout << "Reached the end of the video" << "\n";
                break;
            }

            ExtractFeatures(data.frameIndex);
            // Find matches between feature points of all world points and the current image.
            std::vector<cv::DMatch> world2FrameMatches;
            FindWorldFrameMatches(world2FrameMatches);

            // @TODO remove
            if (detailedPrint) std::cout << "Frame index: " << data.frameIndex << std::endl;

            SFM_Helper::ProcessedFrame &currFrame = SFM.processed_frames[data.frameIndex];
            SFM_Helper::ProcessedFrame &prevFrame = SFM.processed_frames[data.frameIndex - 1];

            std::vector<cv::Point3f> landmarksSubset;
            std::vector<cv::Point2f> featurePointsSubset;
            // Stores the index of each landmark so that we can correctly select the outliers identified by the PnP-RANSAC algorithm.
            std::vector<size_t> landmarkIndices;
            // Find those world points that are referenced in the good feature matches.
            // Find those feature points of the current image that are referenced in the good feature matches.
            for (size_t index = 0; index < world2FrameMatches.size(); ++index)
            {
                const cv::DMatch &match = world2FrameMatches[index];
                const cv::Point3f &worldPoint = data.landmarks[match.queryIdx];
                const cv::Point2f &featurePoint = currFrame.keypoints[match.trainIdx].pt;
                landmarksSubset.push_back(worldPoint);
                featurePointsSubset.push_back(featurePoint);
                landmarkIndices.push_back(match.queryIdx);
            }

            if (detailedPrint) std::cout << "Number of points for PnP: " << landmarksSubset.size() << "\n";
            if (landmarksSubset.size() < 3)
            {
                std::cout << "Not enough correspondences for PnP, need at least 3 3D-2D correspondences. Using 3 correspondences gives multiple solutions of which we just take the first (lowest reporjection error), using more correspondences gives a unique solution (I think)" << "\n";
                break;
            }

            cv::Mat rotation, translation;
            // A list of indices of the inliers in the landmarksSubset and featurePointsSubset.
            cv::Mat pnpInliersIndices;
            cv::solvePnPRansac(landmarksSubset, featurePointsSubset, data.colorIntrinsics, cv::noArray(), rotation, translation, false, 100, 8.0, 0.99, pnpInliersIndices, cv::SOLVEPNP_ITERATIVE);
            if (detailedPrint) std::cout << "Number of inliers after PnP: " << pnpInliersIndices.rows << "\n";

            // Store the just identified inliers in a binary mask for easier access.
            cv::Mat pnpInliers = cv::Mat::zeros(landmarksSubset.size(), 1, CV_8U);
            for (size_t index = 0; index < pnpInliersIndices.rows; ++index)
            {
                const size_t inlierIndex = pnpInliersIndices.at<int>(index);
                pnpInliers.at<uchar>(inlierIndex) = 1;
            }
            for (size_t index = 0; index < landmarksSubset.size(); ++index)
            {
                const size_t landmarkIndex = landmarkIndices[index];
                if (pnpInliers.at<uchar>(index) == 1)
                {
                    // We only count the inliers as seen.
                    data.numSeen[landmarkIndex] += 1;
                }
            }

            cv::Rodrigues(rotation, rotation);
            cv::Mat cameraExtrinsics = ComposeTransformationMatrix(rotation, translation);
            currFrame.projectionMatrix = data.colorIntrinsics * cv::Mat::eye(3, 4, CV_32F) * cameraExtrinsics;
            currFrame.cameraPose = GetInverseTransformation(cameraExtrinsics);

            // Find matches between the previous and the current image.
            std::vector<cv::DMatch> frame2FrameMatches;
            std::vector<cv::Point2f> prevKeypointsFloat;
            FindGoodFeatureMatches(data.frameIndex - 1, data.frameIndex, prevKeypointsFloat, currFrame.keypointsFloat, frame2FrameMatches);
            // Triangulate the matching points.
            cv::Mat homogeneousPoints;
            cv::triangulatePoints(prevFrame.projectionMatrix, currFrame.projectionMatrix, prevKeypointsFloat, currFrame.keypointsFloat, homogeneousPoints);
            std::vector<cv::Point3f> landmarks;
            cv::convertPointsFromHomogeneous(homogeneousPoints.t(), landmarks);

            // Remove existing world points so that we do not add them another time.
            // We know the indices of current frame keypoints used for the world2frame matches and we know the indices of the current frame used for the previous2current frame matches.
            // We store the new world points, the point colors and the feature descriptor indices in new lists here so that we can still properly filter out outliers later on.
            std::vector<cv::Point3f> newLandmarks;
            std::vector<std::tuple<size_t, size_t>> featureDescriptorIndices;
            std::map<size_t, size_t> featureIndex2WorldPointIndex;
            std::vector<Vector4uc> pointColors;
            std::vector<cv::Point2f> newFeaturePoints1;
            std::vector<cv::Point2f> newFeaturePoints2;
            for (size_t matchIndex = 0; matchIndex < world2FrameMatches.size(); ++matchIndex)
            {
                featureIndex2WorldPointIndex[world2FrameMatches[matchIndex].trainIdx] = world2FrameMatches[matchIndex].queryIdx;
            }
            for (size_t matchIndex = 0; matchIndex < frame2FrameMatches.size(); ++matchIndex)
            {
                const size_t featureIndex = frame2FrameMatches[matchIndex].trainIdx;
                if (featureIndex2WorldPointIndex.find(featureIndex) == featureIndex2WorldPointIndex.end())
                {
                    // We did not use this feature point for the world2frame matches, so it is a new point which we should add.
                    newLandmarks.push_back(landmarks[matchIndex]);
                    featureDescriptorIndices.push_back(std::make_tuple(data.frameIndex, featureIndex));
                    // KeypointsFloat are already the subset that we used for triangulation, so we can just use the matchIndex / landmarkIndex here.
                    newFeaturePoints1.push_back(prevKeypointsFloat[matchIndex]);
                    newFeaturePoints2.push_back(currFrame.keypointsFloat[matchIndex]);

                    // Get color of that point
                    const cv::Point2f keypoint = currFrame.keypoints[featureIndex].pt;
                    const cv::Mat &image = currFrame.image;
                    const int x = keypoint.x;
                    const int y = keypoint.y;
                    const unsigned char blue = image.at<cv::Vec4b>(y, x)[0];
                    const unsigned char green = image.at<cv::Vec4b>(y, x)[1];
                    const unsigned char red = image.at<cv::Vec4b>(y, x)[2];
                    Vector4uc color = Vector4uc(red, green, blue, 255);
                    pointColors.push_back(color);

                    if (detailedPrint && featureIndex == 53)
                    {
                        std::cout << "data.frameIndex: " << data.frameIndex << ", " << currFrame.keypoints[featureIndex].pt  << "," << currFrame.keypointsFloat[matchIndex] << std::endl;
                    }
                }
                else
                {
                    // We used this feature point both in the world2frame matches and the frame2frame matches, so this is an existing point, we do not need to add it again. 
                    // We can add its feature descriptor to the list of feature descriptors of this world point, so we know we've seen this point multiple times.
                    const size_t worldPointIndex = featureIndex2WorldPointIndex[featureIndex];
                    if (detailedPrint && featureIndex == 53)
                    {
                        std::cout << "wp index: " << worldPointIndex << std::endl;
                        std::cout << "data.frameIndex: " << data.frameIndex << ", " << currFrame.keypointsFloat[matchIndex] << std::endl;
                    }
                    data.featureDescriptorIndices[worldPointIndex].push_back(std::make_tuple(data.frameIndex, featureIndex));
                }
            }
            if (detailedPrint) std::cout << "Number of new landmarks: " << newLandmarks.size() << "\n";
 
            if (newLandmarks.size() > 0)
            {

                // Important to initialize the inliers mask with all ones.
                cv::Mat inliersMask = cv::Mat::ones(newLandmarks.size(), 1, CV_8U);
                // Filter out point that are behind the camera. Transform points into camera coordinate system and check if z is negative.
                cv::Mat newHomoLandmarks;
                cv::convertPointsToHomogeneous(newLandmarks, newHomoLandmarks);
                newHomoLandmarks = newHomoLandmarks.reshape(1).t();
                cv::Mat cameraPoints = (GetInverseTransformation(currFrame.cameraPose) * newHomoLandmarks);
                for (int index = cameraPoints.cols - 1; index >= 0; --index)
                {
                    const float z = cameraPoints.at<float>(2, index);
                    const float zNorm = z / cameraPoints.at<float>(3, index);
                    if (zNorm < 0) //If the world point lies behind the camera/s
                    {
                        //Set the index of that world point in the inliers mask to 0, i.e. -> label it as an outlier
                        inliersMask.at<uchar>(index) = 0;
                    }
                }
                if (detailedPrint) std::cout << "Number of landmarks behind camera: " << newLandmarks.size() - cv::countNonZero(inliersMask) << "\n";

                cv::Mat extrinsics1 = GetInverseTransformation(prevFrame.cameraPose);
                cv::Mat extrinsics2 = GetInverseTransformation(currFrame.cameraPose);
                
                size_t counter = 0;
                size_t numOutliersRemoved;
                do{
                    numOutliersRemoved = RemoveReprojectionOutliers(newLandmarks, inliersMask, extrinsics1, extrinsics2, newFeaturePoints1, newFeaturePoints2);
                    if (detailedPrint) std::cout << "Number of outliers removed based on reprojection error: " << numOutliersRemoved << "\n";
                    ++counter;
                }while((numOutliersRemoved > 0) && (counter < 3));

                // Actually save all the, now filtered, world points together with their colors and feature descriptor indices.
                for (size_t index = 0; index < featureDescriptorIndices.size(); ++index)
                {
                    if (inliersMask.at<uchar>(index) == 1)
                    {
                        data.landmarks.push_back(newLandmarks[index]);
                        data.pointColors.push_back(pointColors[index]);
                        data.featureDescriptorIndices.push_back(std::vector<std::tuple<size_t, size_t>>());
                        data.featureDescriptorIndices.back().push_back(featureDescriptorIndices[index]);
                        data.numSeen.push_back(2);
                    }
                }

                if (detailedPrint) std::cout << "Number of new (filtered) landmarks: " << cv::countNonZero(inliersMask) << ", total number of landmarks: " << data.landmarks.size() << "\n";
            }
            else
            {
                if (detailedPrint) std::cout << "Number of new (filtered) landmarks: 0, total number of landmarks: " << data.landmarks.size() << "\n";
            }

             // Remove potential outlier landmarks
            if (filter_if_not_seen){
                // Iterate from back to front to not mess up the indices.
                for (int landmarkIndex = data.landmarks.size() - 1; landmarkIndex >= 0; --landmarkIndex)
                {
                    // Remove landmarks that first appeared m frames ago but have been seen less than n times.
                    // There exists always at least one feature descriptor index for each landmark, since otherwise the landmark would not exist.
                    const size_t firstAppearance = std::get<0>(data.featureDescriptorIndices[landmarkIndex][0]);
                    if (((data.frameIndex - firstAppearance) == filter_if_not_seen_n_times_in_m_frames) && (data.numSeen[landmarkIndex] < filter_if_not_seen_n_times))
                    {
                        data.landmarks.erase(data.landmarks.begin() + landmarkIndex);
                        data.featureDescriptorIndices.erase(data.featureDescriptorIndices.begin() + landmarkIndex);
                        data.numSeen.erase(data.numSeen.begin() + landmarkIndex);
                        data.pointColors.erase(data.pointColors.begin() + landmarkIndex);
                    }
                }
            }

            spdlog::info("num landmarks {}", data.landmarks.size());

            // extract observed and predicted landmark locations (2D) to compute reprojection error
            // useful for comparison after bundle adjustment

            auto calculateReprojectionErrorWrapper = [&]() {
                std::vector<cv::Point2f> observed, predicted;
                getObserved(data.landmarks, 
                    data.featureDescriptorIndices, 
                    SFM.processed_frames,
                    observed);
                getPredicted(data.landmarks, 
                    data.featureDescriptorIndices, 
                    SFM.processed_frames,
                    data.colorIntrinsics,
                    predicted); //TODO: rewrite this to just use cv::ProjectPoints
                return calculateReprojectionError(observed, predicted);
            };
            
            auto error_before_ba = calculateReprojectionErrorWrapper();
            if (detailedPrint) std::cout << "mean reprojection error (no ba): " << std::get<0>(error_before_ba) << ", rms: " << std::get<1>(error_before_ba) << "\n";
            reprojectionErrorFs << "no_ba," << i << "," << std::get<0>(error_before_ba) << "," << std::get<1>(error_before_ba) << "\n";
            if (data.config.getRunLocalBundleAdjustment())
            {
                // perform optimization
                runBundleAdjustment(
                    *bundleAdjuster,
                    data.landmarks,
                    SFM.processed_frames,
                    data.featureDescriptorIndices
                );

                // if bundle adjustment performed, compare mean reprojection error before and after BA
                // note: since bundle adjustment occurs for each frame, before BA affected by bundle adjustment from
                //  previous frame. To see true comparision with and without bundle adjustment, see "run_bundle_adjustment" to 
                //  "false" in config.json

                auto error_after_ba = calculateReprojectionErrorWrapper();
                if (detailedPrint) std::cout << "mean before ba: " << std::get<0>(error_before_ba) << ", rms: " << std::get<1>(error_before_ba) << "\n";
                if (detailedPrint) std::cout << "mean after ba: " << std::get<0>(error_after_ba) << ", rms: " << std::get<1>(error_after_ba) << "\n";

                reprojectionErrorFs << "ba," << i << "," << std::get<0>(error_after_ba) << "," << std::get<1>(error_after_ba) << "\n";

                // clear residuals since we iterate over landmarks each time           
                bundleAdjuster->clearAllResiduals();
            }
            auto t2 = std::chrono::high_resolution_clock::now();

             /* Getting number of milliseconds as a double. */
            std::chrono::duration<double, std::milli> timeTaken = t2 - t1;

            if (writeGroundTruth)
            {
                PrintAfterFrameOutput(timeTaken.count());
            }
        }

        if (detailedPrint) std::cout << "Finished estimating points. \n";
    }

    float totalTimeTaken = 0.0f;

    void PrintAfterFrameOutput(double timeTaken)
    {
        data.groundTruthScaleFactor = ComputeScaleFactor();
        //std::cout << "Current Frame IDX is " << data.sensor.GetCurrentFrameCnt() << "\n";
        std::cout << "Frame: " << data.sensor.GetCurrentFrameCnt()+1 << "/" << process_frames << "\n";
        testMetricsOutputFS << data.sensor.GetCurrentFrameCnt() << ",";

        //Print time taken for this frame
        std::cout << "Time taken for current frame: " << timeTaken/1000.0 << " seconds \n";
        testMetricsOutputFS << timeTaken/1000.0 << ",";
        
        //Print total time taken for all frames up until now
        totalTimeTaken += timeTaken;
        std::cout << "Total time taken: " << totalTimeTaken/1000.0 << " seconds \n";
        testMetricsOutputFS << totalTimeTaken/1000.0 << ",";

        //Print number of world points after current frame
        std::cout << "Number of world points: " << data.landmarks.size() << "\n";
        testMetricsOutputFS << data.landmarks.size() << ",";

        //Print current total reprojection error
        auto calculateReprojectionErrorWrapper = [&]() {
                std::vector<cv::Point2f> observed, predicted;
                getObserved(data.landmarks, 
                    data.featureDescriptorIndices, 
                    SFM.processed_frames,
                    observed);
                getPredicted(data.landmarks, 
                    data.featureDescriptorIndices, 
                    SFM.processed_frames,
                    data.colorIntrinsics,
                    predicted); //TODO: rewrite this to just use cv::ProjectPoints
                return calculateReprojectionError(observed, predicted);
            };
            
        auto reprojectionError = calculateReprojectionErrorWrapper();

        std::cout << "RMS Reprojection Error: " << std::get<1>(reprojectionError) << "\n";
        testMetricsOutputFS << std::get<1>(reprojectionError) << ",";

        std::cout << "Reprojection Error: " << std::get<0>(reprojectionError) << "\n";
        testMetricsOutputFS << std::get<0>(reprojectionError) << ",";
        

        //Print current avg reprojection error (total repr. err. / number of world points)
        auto avgReprErr = std::get<0>(reprojectionError);

        std::cout << "Average reprojection error (per world point): " << avgReprErr << "\n";
        testMetricsOutputFS << avgReprErr << ",";

        //Print comparison of all currently reconstructed camera poses to the respective ground truth poses
        CompareToGroundTruth();

        //testMetricsOutputFS << "frame,time,total_time,n_world_points,total_repr_err,avg_rep_err_(per_point)\n";
        //testMetricsOutputFS 
        //    << data.sensor.GetCurrentFrameCnt()+1 << "," 
        //    << timeTaken/1000.0 << ",";
        //    << totalTimeTaken/1000.0 << ","
        //    << data.landmarks.size() << ","
        //    << data.landmarks.size() << ","
        //    << data.landmarks.size() << ",";
        

        //testMetricsOutputFS << "no_ba," << i << "," << std::get<0>(error_before_ba) << "," << std::get<1>(error_before_ba) << "\n";

    }

    void createCameraParamMatrix(double* cameraParamMatrix, cv::Mat cameraPose)
    {
        // rotation: cameraParamMatrix[0-2]
        // translation: cameraParamMatrix[3-5]

        // convert to SO3 (angle-axis rotation)
        cv::Mat rot = cameraPose(cv::Range(0,3), cv::Range(0,3));
        cv::Mat rotVec;
        cv::Rodrigues(rot, rotVec);

        for (size_t i = 0; i < 3; ++i)
        {
            // rotation (SO3)
            cameraParamMatrix[i] = rotVec.at<float>(i);

            // translation
            cameraParamMatrix[i + 3] = cameraPose.at<float>(i, 3);
        }

        for (size_t i = 0; i < 6; ++i)
        {
            spdlog::debug("camera params[{}]: {}", i, cameraParamMatrix[i]);
        }
    }

    void unpackBundleAdjusterParams(BundleAdjuster& bundleAdjuster)
    {
        // unpack results
        const auto observations = bundleAdjuster.getObservations();
        const auto camerasDouble = bundleAdjuster.getCameras();
        const auto worldPointsDouble = bundleAdjuster.getPoints();

        std::set<std::size_t> updatedPoses;
        spdlog::info("Num observations: {}", observations.size());
        for (const auto& observation: observations)
        {
            // spdlog::info(
            //     "Updating frame {}, world point {}", 
            //     observation.cameraId, 
            //     observation.worldPointId);

            // only need to update camera pose once across all observations for given frame
            if (updatedPoses.find(observation.cameraId) == updatedPoses.end())
            {
                spdlog::debug("old camera pose: {}", matToString(SFM.processed_frames[observation.cameraId].cameraPose));

                std::cout << data.colorIntrinsics << std::endl;
                for (size_t j = 0; j < 6; ++j)
                {
                    std::cout << camerasDouble[observation.cameraId * 6 + j] << ",";
                }
                std::cout << "\n";

                // update camera pose
                cv::Vec3f angleAxis(
                    camerasDouble[observation.cameraId * 6], 
                    camerasDouble[observation.cameraId * 6 + 1], 
                    camerasDouble[observation.cameraId * 6 + 2]);
                cv::Vec3f translation(
                    camerasDouble[observation.cameraId * 6 + 3], 
                    camerasDouble[observation.cameraId * 6 + 4], 
                    camerasDouble[observation.cameraId * 6 + 5]);

                // Convert angle-axis to rotation matrix
                cv::Mat rotationMatrix;
                cv::Rodrigues(angleAxis, rotationMatrix);

                // Create the 4x4 pose matrix
                cv::Mat poseMatrix = cv::Mat::eye(4, 4, CV_32F);  // Initialize as identity matrix

                // Copy the rotation matrix to the top-left 3x3 submatrix
                rotationMatrix.copyTo(poseMatrix(cv::Rect(0, 0, 3, 3)));

                // Copy the translation vector to the top-right 3x1 submatrix
                poseMatrix.at<float>(0, 3) = translation[0];
                poseMatrix.at<float>(1, 3) = translation[1];
                poseMatrix.at<float>(2, 3) = translation[2];
                spdlog::debug("new camera pose: {}", matToString(poseMatrix));
                poseMatrix = GetInverseTransformation(poseMatrix);

                SFM.processed_frames[observation.cameraId].cameraPose = poseMatrix;
                

                updatedPoses.insert(observation.cameraId);
            }

            // std::cout << "Now modifying points" << std::endl;
            // std::cout << "camera id: " << observation.cameraId << std::endl;
            spdlog::debug("old world point: {} {} {} camera id: {}, world point id: {}",
                data.landmarks[observation.landmarkId].x,
                data.landmarks[observation.landmarkId].y,
                data.landmarks[observation.landmarkId].z,
                observation.cameraId,
                observation.worldPointId
            );

            data.landmarks[observation.landmarkId].x =
                worldPointsDouble[observation.landmarkId*3];
            data.landmarks[observation.landmarkId].y =
                worldPointsDouble[observation.landmarkId*3+1];
            data.landmarks[observation.landmarkId].z =
                worldPointsDouble[observation.landmarkId*3+2];

            spdlog::debug("new world point: {} {} {} camera id: {} world point id: {}",
                data.landmarks[observation.landmarkId].x,
                data.landmarks[observation.landmarkId].y,
                data.landmarks[observation.landmarkId].z,
                observation.cameraId,
                observation.worldPointId
            );
        }
    }

    // store camera intrinsics in bundle adjuster object
    //  ceres requires double-precision 
    //  note: intrinsics stored as constant (not optimized)
    void packCameraIntrinsics(BundleAdjuster& bundleAdjuster, const cv::Mat& colorIntrinsics)
    {
        float f = colorIntrinsics.at<float>(0, 0);
        float cx = colorIntrinsics.at<float>(0,2);
        float cy = colorIntrinsics.at<float>(1,2);
        double cameraIntrinsicsParam[3] = {
            f, cx, cy
        };

        spdlog::info("camera intrinsics: {}", matToString(data.colorIntrinsics));
        
        bundleAdjuster.addCameraIntrinsicsParam(cameraIntrinsicsParam);
    }

    // store camera extrinsics in bundle adjuster object
    //  ceres requires double-precision
    void packCameraExtrinsics(BundleAdjuster& bundleAdjuster, const std::vector<SFM_Helper::ProcessedFrame>& processed_frames)
    {
        size_t cameraId = 0;
        for (const auto& frame: processed_frames)
        {
            const auto& cameraPose = frame.cameraPose;
            double cameraParam[6];
            createCameraParamMatrix(cameraParam, GetInverseTransformation(cameraPose));
            bundleAdjuster.addCameraParam(cameraParam, cameraId);
            cameraId++;
        }
    }

    // store observations (2d observed points and 3d landmarks) in bundle adjuster object
    //  ceres requires double-precisioin
    //  additionally, add residual to Ceres problem
    void addObservations(BundleAdjuster& bundleAdjuster, 
        const std::vector<cv::Point3f>& landmarks,
        const std::vector<SFM_Helper::ProcessedFrame>& processed_frames,
        const std::vector<std::vector<std::tuple<size_t, size_t>>>& featureDescriptorIndices
    )
    {
        for (size_t j = 0; j < data.landmarks.size(); ++j)
        {
            double worldPoint[3];
            worldPoint[0] = (landmarks[j].x);
            worldPoint[1] = (landmarks[j].y);
            worldPoint[2] = (landmarks[j].z);

            // add worldpoint param, given landmark id
            bundleAdjuster.addWorldPointParam(worldPoint, j);

            // std::cout << "# descriptors: " <<
            // data.featureDescriptorIndices[j].size() << std::endl;

            // for each feature descriptor, add separate observation

            // add observation
            // observations are used to note each observation of world
            // point across frames note, cameraId used to uniquely
            // identify frames/poses k: index of world point in current
            // frame landmark_id: unique 3d point identifier
            for (size_t k = 0;
                    k < featureDescriptorIndices[j].size(); ++k)
            {
                size_t frameId =
                    std::get<0>(featureDescriptorIndices[j][k]);
                size_t descId =
                    std::get<1>(featureDescriptorIndices[j][k]);

                // add observation creates Observation object in bundle adjuster
                // these are primarily used when double-precision data written back out (unpacked)
                // to OpenCV datatypes used by other steps of SFM pipeline
                // we need to know frame idx, feature descriptor idx, and landmark idx
                //  to be able to update data after optimization
                size_t observationIdx = bundleAdjuster.addObservation(
                    frameId, descId, j,
                    processed_frames[frameId]
                        .keypoints[descId]
                        .pt.x,
                    processed_frames[frameId]
                        .keypoints[descId]
                        .pt.y);

                // std::cout << "observed: " <<
                // SFM.processed_frames[frameId].keypoints[descId].pt <<
                // "\n"; std::cout << "keypoints kfloat (frame id: " <<
                // frameId << "): " <<
                // SFM.processed_frames[frameId].keypointsFloat << "\n";

                // add point to problem actually updates the Ceres optimizer
                bundleAdjuster.addPointToProblem(
                    processed_frames[frameId]
                        .keypoints[descId]
                        .pt.x,
                    processed_frames[frameId]
                        .keypoints[descId]
                        .pt.y,
                    bundleAdjuster.mutable_camera_for_observation(
                        frameId),
                    bundleAdjuster.mutable_point_for_observation(j));

                // @TODO: remove following steps
                // only useful if you want to print the projected point values
                cv::Mat rot = GetInverseTransformation(
                    SFM.processed_frames[frameId].cameraPose)(
                    cv::Range(0, 3), cv::Range(0, 3));
                cv::Mat rvec;
                cv::Rodrigues(
                    GetInverseTransformation(
                        SFM.processed_frames[frameId].cameraPose)(
                        cv::Rect(0, 0, 3, 3)),
                    rvec);

                auto p = data.landmarks[j];
                // std::cout << "world point before project: " << p <<
                // std::endl; std::cout << "pose: " <<
                // GetInverseTransformation(SFM.processed_frames[frameId].cameraPose)
                // << std::endl; std::cout << "instrinsics matrix: " <<
                // data.colorIntrinsics << "\n";

                cv::Mat cameraPoint =
                    GetInverseTransformation(
                        SFM.processed_frames[frameId].cameraPose) *
                    (cv::Mat_<float>(4, 1) << p.x, p.y, p.z, 1);
                // std::cout << "camera point: " << cameraPoint <<
                // std::endl;

                cv::Mat projected =
                    data.colorIntrinsics *
                    cameraPoint(cv::Range(0, 3), cv::Range(0, 1));
                projected.at<float>(0) =
                    projected.at<float>(0) / projected.at<float>(2);
                projected.at<float>(1) =
                    projected.at<float>(1) / projected.at<float>(2);

                cv::Mat translate = GetInverseTransformation(
                    SFM.processed_frames[frameId].cameraPose)(
                    cv::Range(0, 3), cv::Range(3, 4));
                cv::Mat reprojectedPoints;
                cv::projectPoints(
                    data.landmarks, rvec,
                    GetInverseTransformation(
                        SFM.processed_frames[frameId].cameraPose)(
                        cv::Rect(3, 0, 1, 3)),
                    data.colorIntrinsics, cv::noArray(),
                    reprojectedPoints);

                // std::cout << "landmark: " << data.landmarks[j] <<
                // std::endl; std::cout << "projected opencv: " <<
                // reprojectedPoints.at<cv::Point2f>(j) << std::endl;
            }
        }
    }

    void runBundleAdjustment(
        BundleAdjuster& bundleAdjuster, 
        const std::vector<cv::Point3f>& landmarks,
        const std::vector<SFM_Helper::ProcessedFrame>& processed_frames,
        const std::vector<std::vector<std::tuple<size_t, size_t>>>& featureDescriptorIndices) 
    { 
        // spdlog::info("Running bundle adjustment solver");

        // create double-precision storage for camera extrinsics
        packCameraExtrinsics(bundleAdjuster, processed_frames);

        // for each landmark, loop thru associated feature descriptors and add residuals
        addObservations(bundleAdjuster, landmarks, processed_frames, featureDescriptorIndices);

        // solve optimization problem
        bundleAdjuster.solve(); 

        // unpack/write double-precision storage from BA
        //   back to OpenCV types used by SFM pipeline
        unpackBundleAdjusterParams(bundleAdjuster);
    }

    void VisualiseSFMInitResults()
    {
        if (detailedPrint) std::cout << "Visualising SFM Init Results..." << std::endl;
        SimpleMesh geometry;
        std::ofstream worldPointsCsv;
        worldPointsCsv.open("worldPoints.csv", std::ios::trunc);

        for (size_t pointIndex = 0; pointIndex < data.landmarks.size(); ++pointIndex)
        {
            const cv::Point3f worldPoint = data.landmarks[pointIndex];
            Vertex vertex;
            Eigen::Vector4f point;
            point(0) = worldPoint.x;
            point(1) = worldPoint.y;
            point(2) = worldPoint.z;
            point(3) = 1.0f;
            worldPointsCsv << point(0) << "," << point(1) << "," << point(2) << "," << point(3) << ",";
            // std::cout << "position: " << point << std::endl;
            vertex.position = point;
            vertex.color = data.pointColors[pointIndex];
            worldPointsCsv << (int)vertex.color(0) << "," << (int)vertex.color(1) << "," << (int)vertex.color(2) << "," << (int)vertex.color(3) << "\n";
            // std::cout << "color: " << vertex.color << std::endl;
            geometry.addVertex(vertex);
        }

        worldPointsCsv.close();
        
        SimpleMesh camerasMesh;
        // Slowly change color from red to blue to indicate the order of the camera poses
        float blue = 255.0f / (SFM.processed_frames.size() - 1);
        for (int poseIndex = 0; poseIndex < SFM.processed_frames.size(); ++poseIndex)
        {
            // Don't ask me why, but we need to pass the inverse of the camera pose for it to render the actual camera pose.
            const cv::Mat cameraPose = GetInverseTransformation(SFM.processed_frames[poseIndex].cameraPose);
            Eigen::Matrix4f eigenCameraPose;
            cv::cv2eigen(cameraPose, eigenCameraPose);

            Vector4uc color(255 - int(blue * poseIndex), 0, int(blue * poseIndex), 255);
            SimpleMesh cameraMesh = SimpleMesh::camera(eigenCameraPose, 0.003f, color);
            camerasMesh = SimpleMesh::joinMeshes(cameraMesh, camerasMesh);
        }

        if (writeGroundTruth)
        {
            cv::Mat firstEstimatedPose = SFM.processed_frames[0].cameraPose;
            cv::Mat firstTruthPose = data.groundTruthPoses[0];
            
            if (config_groundTruthScaleFactor < 0.0f) // If the scale factor is not manually set in config
            {

                float medianScaleFactor;
                float meanScaleFactor;
                std::vector<float> scaleFactors;
                // Get the scale factors for each frame.
                float totalScaleFactor = 0.0f;

                for (size_t poseIndex = 1; poseIndex < data.groundTruthPoses.size(); ++poseIndex)
                {
                    const cv::Mat groundTruthPose = data.groundTruthPoses[poseIndex];
                    const cv::Mat estimatedPose = SFM.processed_frames[poseIndex].cameraPose;
                    const cv::Mat deltaGroundTruth = groundTruthPose * GetInverseTransformation(firstTruthPose);
                    const cv::Mat deltaEstimated = estimatedPose * GetInverseTransformation(firstEstimatedPose);

                    const float scaleFactor = cv::norm(deltaEstimated(cv::Rect(3, 0, 1, 3))) / cv::norm(deltaGroundTruth(cv::Rect(3, 0, 1, 3)));
                    scaleFactors.push_back(scaleFactor);
                    totalScaleFactor += scaleFactor;
                }

                // Use the median scale factor to scale the estimated poses.
                std::sort(scaleFactors.begin(), scaleFactors.end());
                medianScaleFactor = scaleFactors[scaleFactors.size() / 2];
                meanScaleFactor = totalScaleFactor / scaleFactors.size();
                const float optimalScaleFactor = ComputeScaleFactor();
                if (detailedPrint) 
                {
                    std::cout << "Median scale factor is " << medianScaleFactor << "\n";
                    std::cout << "Mean scale factor is " << meanScaleFactor << "\n";
                    std::cout << "Optimal scale factor is " << optimalScaleFactor << "\n";
                }

                if (scaleFactorType == 0)
                {
                    data.groundTruthScaleFactor = medianScaleFactor;
                }
                else if (scaleFactorType == 1)
                {
                    data.groundTruthScaleFactor = meanScaleFactor;
                }
                else if (scaleFactorType == 2)
                {
                    data.groundTruthScaleFactor = optimalScaleFactor;
                }
            }
            else
            {
                data.groundTruthScaleFactor = config_groundTruthScaleFactor;
            }

            // Write aligned ground truth poses to output mesh.
             // Slowly change color from light green to dark green to indicate the order of the camera poses
            float green = 255.0f / (data.groundTruthPoses.size() - 1);
            for (size_t poseIndex = 0; poseIndex < data.groundTruthPoses.size(); ++poseIndex)
            {
                // Relative to the first ground truth pose.
                cv::Mat relativeGroundTruth = data.groundTruthPoses[poseIndex] * GetInverseTransformation(firstTruthPose);
                // Scale up the relative position / movement (only the translational components) by the scale factor so it should overall align with the estimated poses.
                relativeGroundTruth.at<float>(0, 3) *= data.groundTruthScaleFactor;
                relativeGroundTruth.at<float>(1, 3) *= data.groundTruthScaleFactor;
                relativeGroundTruth.at<float>(2, 3) *= data.groundTruthScaleFactor;
                // Add the relative ground truth position on top of the first estimated camera pose so that both start at the same position.
                cv::Mat cameraPose = relativeGroundTruth * GetInverseTransformation(firstEstimatedPose);

                Eigen::Matrix4f eigenCameraPose;
                cv::cv2eigen(cameraPose, eigenCameraPose);
                
                Vector4uc color(0, 255 - int(green * poseIndex), 0, 255);
                SimpleMesh cameraMesh = SimpleMesh::camera(eigenCameraPose, 0.003f, color);
                camerasMesh = SimpleMesh::joinMeshes(cameraMesh, camerasMesh);
            }
        }

        SimpleMesh mesh = SimpleMesh::joinMeshes(geometry, camerasMesh);
        std::cout << "Writing mesh..." << std::endl;
        mesh.writeMesh(datasetName + "/mesh_" + GetConfigOptionsString() + ".off");
	}

    float ComputeScaleFactor()
    {
        cv::Mat firstEstimatedPose = GetInverseTransformation(SFM.processed_frames[0].cameraPose);
        cv::Mat firstTruthPose = data.groundTruthPoses[0];

        float totalDistanceGroundTruth = 0.0f;
        for (size_t poseIndex = 0; poseIndex < data.groundTruthPoses.size(); ++poseIndex)
        {
            const cv::Mat cameraPose = data.groundTruthPoses[poseIndex];
            const cv::Mat cameraPoseFromOrigin = cameraPose * GetInverseTransformation(firstTruthPose);
            totalDistanceGroundTruth += cv::norm(cameraPoseFromOrigin(cv::Rect(3, 0, 1, 3)));
        }
        const float avgDistanceGroundTruth = totalDistanceGroundTruth / data.groundTruthPoses.size();

        float totalDistanceEstimated = 0.0f;
        for (size_t poseIndex = 0; poseIndex < SFM.processed_frames.size(); ++poseIndex)
        {
            const cv::Mat cameraPose = GetInverseTransformation(SFM.processed_frames[poseIndex].cameraPose);
            const cv::Mat cameraPoseFromOrigin = cameraPose * GetInverseTransformation(firstEstimatedPose);
            totalDistanceEstimated += cv::norm(cameraPoseFromOrigin(cv::Rect(3, 0, 1, 3)));
        }
        const float avgDistanceEstimated = totalDistanceEstimated / SFM.processed_frames.size();
        const float scaleFactor = avgDistanceEstimated / avgDistanceGroundTruth;

        return scaleFactor;
    }

    SimpleMesh WriteCameraPoses()
    {
        SimpleMesh camerasMesh;
        cv::Mat firstEstimatedPose = GetInverseTransformation(SFM.processed_frames[0].cameraPose);
        cv::Mat firstTruthPose = data.groundTruthPoses[0];

        const float blue = 255.0f / (SFM.processed_frames.size() - 1);
        for (size_t poseIndex = 0; poseIndex < SFM.processed_frames.size(); ++poseIndex)
        {
            const cv::Mat cameraPose = GetInverseTransformation(SFM.processed_frames[poseIndex].cameraPose);
            const cv::Mat cameraPoseFromOrigin = cameraPose * GetInverseTransformation(firstEstimatedPose);

            Eigen::Matrix4f eigenCameraPose;
            cv::cv2eigen(cameraPoseFromOrigin, eigenCameraPose);
            const SimpleMesh cameraMesh = SimpleMesh::camera(eigenCameraPose, 0.003f, Vector4uc(255 - int(blue * poseIndex), 0, int(blue * poseIndex), 255));
            camerasMesh = SimpleMesh::joinMeshes(camerasMesh, cameraMesh);
        }

        
        const float scaleFactor = ComputeScaleFactor();
        data.groundTruthScaleFactor = scaleFactor;
        std::cout << "Scale factor: " << scaleFactor << "\n";

        const float green = 255.0f / (SFM.processed_frames.size() - 1);
        for (size_t poseIndex = 0; poseIndex < data.groundTruthPoses.size(); ++poseIndex)
        {
            const cv::Mat cameraPose = data.groundTruthPoses[poseIndex];
            // Pose relative to the first ground truth pose.
            const cv::Mat originPose = cameraPose * GetInverseTransformation(firstTruthPose);
            cv::Mat scaledPose = originPose;
            scaledPose.at<float>(0, 3) *= scaleFactor;
            scaledPose.at<float>(1, 3) *= scaleFactor;
            scaledPose.at<float>(2, 3) *= scaleFactor;
            std::cout << "Origin Pose location: (" << originPose.at<float>(0, 3) << ", " << originPose.at<float>(1, 3) << ", " << originPose.at<float>(2, 3) << ")\n";
            const cv::Mat alignedPose = scaledPose * firstEstimatedPose;

            Eigen::Matrix4f eigenCameraPose;
            cv::cv2eigen(alignedPose, eigenCameraPose);
            const SimpleMesh cameraMesh = SimpleMesh::camera(eigenCameraPose, 0.003f, Vector4uc(0, 255 - int(green * poseIndex), 0, 255));
            camerasMesh = SimpleMesh::joinMeshes(camerasMesh, cameraMesh);
        }

        return camerasMesh;
    }

    void CompareToGroundTruth()
    {
        cv::Mat firstEstimatedPose = SFM.processed_frames[0].cameraPose;
        cv::Mat firstTruthPose = data.groundTruthPoses[0];
            
        //Do the comparison between estimated and ground truth trajectories
        float totalError_tx = 0.0f,
            totalError_ty = 0.0f,
            totalError_tz = 0.0f;
        float meanError_tx = 0.0f,
            meanError_ty = 0.0f,
            meanError_tz = 0.0f;
        float minError_tx = std::numeric_limits<float>::max(),
            minError_ty = std::numeric_limits<float>::max(),
            minError_tz = std::numeric_limits<float>::max();
        float maxError_tx = std::numeric_limits<float>::min(),
            maxError_ty = std::numeric_limits<float>::min(),
            maxError_tz = std::numeric_limits<float>::min();
        float stdError_tx = 0.0f,
            stdError_ty = 0.0f,
            stdError_tz = 0.0f;

        float totalError_rx = 0.0f,
            totalError_ry = 0.0f,
            totalError_rz = 0.0f;
        float meanError_rx = 0.0f,
            meanError_ry = 0.0f,
            meanError_rz = 0.0f;
        float minError_rx = std::numeric_limits<float>::max(),
            minError_ry = std::numeric_limits<float>::max(),
            minError_rz = std::numeric_limits<float>::max();
        float maxError_rx = std::numeric_limits<float>::min(),
            maxError_ry = std::numeric_limits<float>::min(),
            maxError_rz = std::numeric_limits<float>::min();
        float stdError_rx = 0.0f,
            stdError_ry = 0.0f,
            stdError_rz = 0.0f;
            
        bool printDebug = false;

        if (printDebug) std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n";
        //Skip over the first frame for both estimated and ground truth, as these are assumed to be origin
        for (int poseIndex = 1; poseIndex < SFM.processed_frames.size(); ++poseIndex)
        {
            // Transform the estimated pose in such way that its origin is in the world origin. (It looks wrong because the estimated poses are actually inversed).
            cv::Mat pose = GetInverseTransformation(SFM.processed_frames[poseIndex].cameraPose) * firstEstimatedPose;
            cv::Vec3f t_est(pose.at<float>(0,3), pose.at<float>(1,3), pose.at<float>(2,3));
            cv::Mat a1 = pose(cv::Rect(0, 0, 3, 3));
            Eigen::Matrix3f b1;
            cv::cv2eigen(a1, b1);
            Eigen::Quaternionf r_est(b1);
            auto euler_est = b1.eulerAngles(2, 1, 0);

            // Transform the ground truth pose in such way that its origin is in the world origin.
            pose = data.groundTruthPoses[poseIndex] * GetInverseTransformation(firstTruthPose);
            cv::Vec3f t_truth(
                pose.at<float>(0,3) * data.groundTruthScaleFactor, 
                pose.at<float>(1,3) * data.groundTruthScaleFactor, 
                pose.at<float>(2,3) * data.groundTruthScaleFactor);
            cv::Mat a = pose(cv::Rect(0, 0, 3, 3));
            Eigen::Matrix3f b;
            cv::cv2eigen(a, b);
            Eigen::Quaternionf r_truth(b);
            auto euler_truth = b.eulerAngles(2, 1, 0);

            //Absolute t error
            cv::Vec3f err_t(abs((t_est - t_truth)[0]), abs((t_est - t_truth)[1]), abs((t_est - t_truth)[2]));
            err_t /= data.groundTruthScaleFactor;
                
            //Absolute r error
            cv::Vec3f err_r(
                GetRadiansError(euler_est[0], euler_truth[0]),
                GetRadiansError(euler_est[1], euler_truth[1]),
                GetRadiansError(euler_est[2], euler_truth[2])
            );
                
            if (printDebug)
            {
                std::cout << "Calculating errors between estimated and groundtruth poses. poseIndex is " << poseIndex << ", \n";
                std::cout << "Est pose t is " << t_est << "\n";
                std::cout << "Est pose r is " << r_est << "\n";
                std::cout << "Est pose r euler angles in XYZ is " << euler_est << "\n"; 
                std::cout << "Ground truth pose t is " << t_truth << "\n";
                std::cout << "Ground truth pose r is " << r_truth << "\n"; //This is correct
                std::cout << "Ground truth pose r euler angles in XYZ is " << euler_truth << "\n"; 
                std::cout << "Pose t err is " << err_t << "\n";
                std::cout << "Pose r err is " << err_r << "\n";
                std::cout << "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n";
            }

            totalError_tx += err_t[0];
            totalError_ty += err_t[1];
            totalError_tz += err_t[2];

            if (err_t[0] < minError_tx) minError_tx = err_t[0];
            if (err_t[1] < minError_ty) minError_ty = err_t[1];
            if (err_t[2] < minError_tz) minError_tz = err_t[2];

            if (err_t[0] > maxError_tx) maxError_tx = err_t[0];
            if (err_t[1] > maxError_ty) maxError_ty = err_t[1];
            if (err_t[2] > maxError_tz) maxError_tz = err_t[2];

            totalError_rx += err_r[0];
            totalError_ry += err_r[1];
            totalError_rz += err_r[2];

            if (err_r[0] < minError_rx) minError_rx = err_r[0];
            if (err_r[1] < minError_ry) minError_ry = err_r[1];
            if (err_r[2] < minError_rz) minError_rz = err_r[2];

            if (err_r[0] > maxError_rx) maxError_rx = err_r[0];
            if (err_r[1] > maxError_ry) maxError_ry = err_r[1];
            if (err_r[2] > maxError_rz) maxError_rz = err_r[2];
        }

        meanError_tx = totalError_tx / (SFM.processed_frames.size() - 1);
        meanError_ty = totalError_ty / (SFM.processed_frames.size() - 1);
        meanError_tz = totalError_tz / (SFM.processed_frames.size() - 1);

        meanError_rx = totalError_rx / (SFM.processed_frames.size() - 1);
        meanError_ry = totalError_ry / (SFM.processed_frames.size() - 1);
        meanError_rz = totalError_rz / (SFM.processed_frames.size() - 1);
        //meanError_rw = totalError_rw / (SFM.processed_frames.size() - 1);

        float stdTotalDist_tx = 0.0f,
            stdTotalDist_ty = 0.0f,
            stdTotalDist_tz = 0.0f;

        float stdTotalDist_rx = 0.0f,
            stdTotalDist_ry = 0.0f,
            stdTotalDist_rz = 0.0f,
            stdTotalDist_rw = 0.0f;

        //Skip over the first frame for both estimated and ground truth, as these are assumed to be origin
        for (int poseIndex = 1; poseIndex < SFM.processed_frames.size(); ++poseIndex)
        {
            // Transform the estimated pose in such way that its origin is in the world origin. (It looks wrong because the estimated poses are actually inversed).
            cv::Mat pose = GetInverseTransformation(SFM.processed_frames[poseIndex].cameraPose) * firstEstimatedPose;
            cv::Vec3f t_est(pose.at<float>(0,3), pose.at<float>(1,3), pose.at<float>(2,3));
            cv::Mat a1 = pose(cv::Rect(0, 0, 3, 3));
            Eigen::Matrix3f b1;
            cv::cv2eigen(a1, b1);
            Eigen::Quaternionf r_est(b1);
            auto euler_est = b1.eulerAngles(2, 1, 0);

            // Transform the ground truth pose in such way that its origin is in the world origin.
            pose = data.groundTruthPoses[poseIndex] * GetInverseTransformation(firstTruthPose);
            cv::Vec3f t_truth(
                pose.at<float>(0,3) * data.groundTruthScaleFactor, 
                pose.at<float>(1,3) * data.groundTruthScaleFactor, 
                pose.at<float>(2,3) * data.groundTruthScaleFactor);
            cv::Mat a = pose(cv::Rect(0, 0, 3, 3));
            Eigen::Matrix3f b;
            cv::cv2eigen(a, b);
            Eigen::Quaternionf r_truth(b);
            auto euler_truth = b.eulerAngles(2, 1, 0);

            //Absolute t error
            cv::Vec3f err_t(abs((t_est - t_truth)[0]), abs((t_est - t_truth)[1]), abs((t_est - t_truth)[2]));
            err_t /= data.groundTruthScaleFactor;

            //Absolute r error
            cv::Vec3f err_r(
                GetRadiansError(euler_est[0], euler_truth[0]), 
                GetRadiansError(euler_est[1], euler_truth[1]),
                GetRadiansError(euler_est[2], euler_truth[2])
            );

            //Calc standard deviation of error from mean
            stdTotalDist_tx += powf(err_t[0] - meanError_tx, 2.0f);
            stdTotalDist_ty += powf(err_t[1] - meanError_ty, 2.0f);
            stdTotalDist_tz += powf(err_t[2] - meanError_tz, 2.0f);

            stdTotalDist_rx += powf(err_r[0] - meanError_rx, 2.0f);
            stdTotalDist_ry += powf(err_r[1] - meanError_ry, 2.0f);
            stdTotalDist_rz += powf(err_r[2] - meanError_rz, 2.0f);
        }

        stdError_tx = sqrt(stdTotalDist_tx / (SFM.processed_frames.size() - 1));
        stdError_ty = sqrt(stdTotalDist_ty / (SFM.processed_frames.size() - 1));
        stdError_tz = sqrt(stdTotalDist_tz / (SFM.processed_frames.size() - 1));

        stdError_rx = sqrt(stdTotalDist_rx / (SFM.processed_frames.size() - 1));
        stdError_ry = sqrt(stdTotalDist_ry / (SFM.processed_frames.size() - 1));
        stdError_rz = sqrt(stdTotalDist_rz / (SFM.processed_frames.size() - 1));

        //std::size_t found = datasetPath.find_last_of("/");
		//std::size_t found2 = datasetPath.substr(found + 1).find_last_of(".");
		//std::string pathName = datasetPath.substr(found + 1).substr(0, found2);

        //std::stringstream filename;
        //filename << datasetName << "/Comparison_To_Groundtruth_ " << GetConfigOptionsString() << ".txt";
        //std::cout << "Writing evaluation to file: '" << filename.str() << "'\n";

        //std::ofstream fileWr;
        //fileWr.open(filename.str(), std::ios_base::app);
        //if (fileWr.is_open())
        //{
        //    fileWr << "Results of comparison to dataset at path: " << datasetPath << "\n";
        //    fileWr << "----------------------------------------\n";
        //    fileWr << "Mean T_x error: " << meanError_tx << "\n";
        //    fileWr << "Mean T_y error: " << meanError_ty << "\n";
        //    fileWr << "Mean T_z error: " << meanError_tz << "\n";
        //    fileWr << "STD of T_x error: " << stdError_tx << "\n";
        //    fileWr << "STD of T_y error: " << stdError_ty << "\n";
        //    fileWr << "STD of T_z error: " << stdError_tz << "\n";
        //    fileWr << "Min T_x error:" << minError_tx << "\n";
        //    fileWr << "Min T_y error: " << minError_ty << "\n";
        //    fileWr << "Min T_z error: " << minError_tz << "\n";
        //    fileWr << "Max T_x error: " << maxError_tx << "\n";
        //    fileWr << "Max T_y error: " << maxError_ty << "\n";
        //    fileWr << "Max T_z error: " << maxError_tz << "\n";
        //    fileWr << "----------------------------------------\n";
        //    fileWr << "Mean R_x error: " << meanError_rx << "\n";
        //    fileWr << "Mean R_y error: " << meanError_ry << "\n";
        //    fileWr << "Mean R_z error: " << meanError_rz << "\n";
        //    fileWr << "STD of R_x error: " << stdError_rx << "\n";
        //    fileWr << "STD of R_y error: " << stdError_ry << "\n";
        //    fileWr << "STD of R_z error: " << stdError_rz << "\n";
        //    fileWr << "Min R_x error: " << minError_rx << "\n";
        //    fileWr << "Min R_y error: " << minError_ry << "\n";
        //    fileWr << "Min R_z error: " << minError_rz << "\n";
        //    fileWr << "Max R_x error: " << maxError_rx << "\n";
        //    fileWr << "Max R_y error: " << maxError_ry << "\n";
        //    fileWr << "Max R_z error: " << maxError_rz << "\n";
        //    fileWr << "----------------------------------------\n";


        //    testMetricsOutputFS << meanError_tx << ",";
        //    testMetricsOutputFS << meanError_ty << ",";
        //    testMetricsOutputFS << meanError_tz << ",";
        //    testMetricsOutputFS << stdError_tx << ",";
        //    testMetricsOutputFS << stdError_ty << ",";
        //    testMetricsOutputFS << stdError_tz << ",";
        //    testMetricsOutputFS << minError_tx << ",";
        //    testMetricsOutputFS << minError_ty << ",";
        //    testMetricsOutputFS << minError_tz << ",";
        //    testMetricsOutputFS << maxError_tx << ",";
        //    testMetricsOutputFS << maxError_ty << ",";
        //    testMetricsOutputFS << maxError_tz << ",";

        //    testMetricsOutputFS << meanError_rx << ",";
        //    testMetricsOutputFS << meanError_ry << ",";
        //    testMetricsOutputFS << meanError_rz << ",";
        //    testMetricsOutputFS << stdError_rx << ",";
        //    testMetricsOutputFS << stdError_ry << ",";
        //    testMetricsOutputFS << stdError_rz << ",";
        //    testMetricsOutputFS << minError_rx << ",";
        //    testMetricsOutputFS << minError_ry << ",";
        //    testMetricsOutputFS << minError_rz << ",";
        //    testMetricsOutputFS << maxError_rx << ",";
        //    testMetricsOutputFS << maxError_ry << ",";
        //    testMetricsOutputFS << maxError_rz << ",";




        //    //testMetricsOutputFS << "frame,time,total_time,n_world_points,total_repr_err,avg_rep_err_(per_point),cam_mean_t_x_err,cam_mean_t_y_err,cam_mean_t_z_err,cam_std_t_x_err,cam_std_t_y_err,cam_std_t_z_err,cam_min_t_x_err,cam_min_t_y_err,cam_min_t_z_err,cam_max_t_x_err,cam_max_t_y_err,cam_max_t_z_err,cam_mean_r_x_err,cam_mean_r_y_err,cam_mean_r_z_err,cam_std_r_x_err,cam_std_r_y_err,cam_std_r_z_err,cam_min_r_x_err,cam_min_r_y_err,cam_min_r_z_err,cam_max_r_x_err,cam_max_r_y_err,cam_max_r_z_err \n";

        //}
        //    
        //fileWr.close();

        testMetricsOutputFS << meanError_tx << ",";
        testMetricsOutputFS << meanError_ty << ",";
        testMetricsOutputFS << meanError_tz << ",";
        testMetricsOutputFS << stdError_tx << ",";
        testMetricsOutputFS << stdError_ty << ",";
        testMetricsOutputFS << stdError_tz << ",";
        testMetricsOutputFS << minError_tx << ",";
        testMetricsOutputFS << minError_ty << ",";
        testMetricsOutputFS << minError_tz << ",";
        testMetricsOutputFS << maxError_tx << ",";
        testMetricsOutputFS << maxError_ty << ",";
        testMetricsOutputFS << maxError_tz << ",";

        testMetricsOutputFS << meanError_rx << ",";
        testMetricsOutputFS << meanError_ry << ",";
        testMetricsOutputFS << meanError_rz << ",";
        testMetricsOutputFS << stdError_rx << ",";
        testMetricsOutputFS << stdError_ry << ",";
        testMetricsOutputFS << stdError_rz << ",";
        testMetricsOutputFS << minError_rx << ",";
        testMetricsOutputFS << minError_ry << ",";
        testMetricsOutputFS << minError_rz << ",";
        testMetricsOutputFS << maxError_rx << ",";
        testMetricsOutputFS << maxError_ry << ",";
        testMetricsOutputFS << maxError_rz << "\n";



        std::cout << "-------------------------\n";
        std::cout << "Mean X_T error is " << meanError_tx << "\n";
        std::cout << "Mean Y_T error is " << meanError_ty << "\n";
        std::cout << "Mean Z_T error is " << meanError_tz << "\n";
        std::cout << "STD of X_T error is " << stdError_tx << "\n";
        std::cout << "STD of Y_T error is " << stdError_ty << "\n";
        std::cout << "STD of Z_T error is " << stdError_tz << "\n";
        std::cout << "Min X_T error is " << minError_tx << "\n";
        std::cout << "Min Y_T error is " << minError_ty << "\n";
        std::cout << "Min Z_T error is " << minError_tz << "\n";
        std::cout << "Max X_T error is " << maxError_tx << "\n";
        std::cout << "Max Y_T error is " << maxError_ty << "\n";
        std::cout << "Max Z_T error is " << maxError_tz << "\n";
        std::cout << "-------------------------\n";
        std::cout << "Mean X_R error is " << meanError_rx << "\n";
        std::cout << "Mean Y_R error is " << meanError_ry << "\n";
        std::cout << "Mean Z_R error is " << meanError_rz << "\n";
        std::cout << "STD of X_R error is " << stdError_rx << "\n";
        std::cout << "STD of Y_R error is " << stdError_ry << "\n";
        std::cout << "STD of Z_R error is " << stdError_rz << "\n";
        std::cout << "Min X_R error is " << minError_rx << "\n";
        std::cout << "Min Y_R error is " << minError_ry << "\n";
        std::cout << "Min Z_R error is " << minError_rz << "\n";
        std::cout << "Max X_R error is " << maxError_rx << "\n";
        std::cout << "Max Y_R error is " << maxError_ry << "\n";
        std::cout << "Max Z_R error is " << maxError_rz << "\n";
        std::cout << "-------------------------\n";
        
    }

	void DebugDisplay(int frameIndex = -1) 
	{
		if (!data.config.getRunHeadless())
		{
			namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
			cv::imshow("Display Image", DebugOutputImage);
			cv::waitKey(0);
		}
		else 
		{
			if (frameIndex == -1) 
			{
				cv::imwrite("out.png", DebugOutputImage);
			}
			else
			{
				// cv::imwrite("match_" + std::to_string(frameIndex) + ".png", DebugOutputImage);
			}
		}
	}

    void testBlurFiltering()
    {
        std::filesystem::remove_all("blurry");
        std::filesystem::remove_all("non_blurry");
        std::filesystem::create_directory("blurry");
        std::filesystem::create_directory("non_blurry");

        int frameIndex = 0;
        int numBlurry = 0;
        int numNonBlurry = 0;
        while(data.sensor.ProcessNextFrame())
        {
            float variance;
            cv::Mat bufferFrame(data.sensor.GetColorImageHeight(), data.sensor.GetColorImageWidth(), CV_8UC4, data.sensor.GetColorRGBX());
            cv::Mat frame = bufferFrame.clone();
            const bool blurry = isBlurryImage(frame, variance);

            if (blurry)
            {
                // Save to blurry folder
                cv::imwrite("blurry/frame_" + std::to_string(frameIndex) + "_var_" + std::to_string(variance) + ".png", frame);
                ++numBlurry; 
            }
            else 
            {
                // Save to non-blurry folder
                cv::imwrite("non_blurry/frame_" + std::to_string(frameIndex) + "_var_" + std::to_string(variance) + ".png", frame);
                ++numNonBlurry;
            }

            ++frameIndex;
        }

        std::cout << "Num blurry: " << numBlurry << ", num non-blurry: " << numNonBlurry << "\n";
    }

    std::string GetConfigOptionsString()
    {
        std::stringstream filename;
        filename << process_frames << "f_";
        filename << std::fixed << std::setprecision(2) << match_ratio_thresh << "mth_";
        filename << frameInterval << "fi_";
        filename << (data.config.getRunLocalBundleAdjustment() ? "localba_" : "nolocalba_");
        filename << (data.config.getRunGlobalBundleAdjustment() ? "globalba_" : "noglobalba_");
        if (filter_if_not_seen)
        {
            filename << "fins_" << filter_if_not_seen_n_times << "_in_" << filter_if_not_seen_n_times_in_m_frames << "_";
        }
        else
        {
            filename << "nofins_";
        }
        filename << min_reprojection_error_threshold << "mret_";
        filename << reprojection_error_max_std_devs << "remsd";
        filename << "_" << data.config.getBlurryThreshold();
        filename << "_" + data.config.getLossFunctionType();

        return filename.str();
    }

    // Returns absolute error between two angles in radians.
    float GetRadiansError(const float angle1, const float angle2)
    {
        float error = angle1 - angle2;
        error = fmod(abs(error + M_PI), 2 * M_PI) - M_PI;
        return abs(error);
    }

    void PerformHardFiltering()
    {
        // Remove all points that we have not seen at least 3 times in total.
        // Iterate from back to front to not mess up the indices.
        size_t numLandmarksBefore = data.landmarks.size();
        for (int landmarkIndex = data.landmarks.size() - 1; landmarkIndex >= 0; --landmarkIndex)
        {
            if (data.numSeen[landmarkIndex] < 3)
            {
                data.landmarks.erase(data.landmarks.begin() + landmarkIndex);
                data.featureDescriptorIndices.erase(data.featureDescriptorIndices.begin() + landmarkIndex);
                data.numSeen.erase(data.numSeen.begin() + landmarkIndex);
                data.pointColors.erase(data.pointColors.begin() + landmarkIndex);
            }
        }

        std::cout << "Removed " << numLandmarksBefore - data.landmarks.size() << " points that were seen not enough times.\n";
    }
};

int main(int argc, char** argv)
{
	StructureFromMotion sfm;



    //Read in all possible values for a parameter for 

	if (!sfm.Init()) return -1;
	
	if (!sfm.InitialPoseEstimate())
    {
        std::cout << "Failed to estimate initial pose" << "\n";
        return -1;
    }
	sfm.InitialPoseEsimateMainLoop();



    sfm.PerformHardFiltering();

    //TODO: Print last after frame data out here (Or maybe not)
    if (sfm.writeGroundTruth)
    {
        sfm.PrintAfterFrameOutput(0.0);
    }

    if (sfm.data.config.getRunGlobalBundleAdjustment())
    {
        std::cout << "Running BA\n";
        auto calculateReprojectionErrorWrapper = [&]() {
            std::vector<cv::Point2f> observed, predicted;
            sfm.getObserved(sfm.data.landmarks, 
                sfm.data.featureDescriptorIndices, 
                sfm.SFM.processed_frames,
                observed);
            sfm.getPredicted(sfm.data.landmarks, 
                sfm.data.featureDescriptorIndices, 
                sfm.SFM.processed_frames,
                sfm.data.colorIntrinsics,
                predicted);
            return calculateReprojectionError(observed, predicted);
        };
        
        auto error_before_ba = calculateReprojectionErrorWrapper();
        if (detailedPrint) std::cout << "mean reprojection error (no ba): " << std::get<0>(error_before_ba) << ", rms: " << std::get<1>(error_before_ba) << "\n";
        // sfm.reprojectionErrorFs << "no_ba," << 100<< "," << std::get<0>(error_before_ba) << "," << std::get<1>(error_before_ba) << "\n";

        // add back if you want global bundle adjustment step
        sfm.runBundleAdjustment(
            *sfm.bundleAdjuster,
            sfm.data.landmarks,
            sfm.SFM.processed_frames,
            sfm.data.featureDescriptorIndices
        );

        auto error_after_ba = calculateReprojectionErrorWrapper();
        if (detailedPrint) std::cout << "mean before ba: " << std::get<0>(error_before_ba) << ", rms: " << std::get<1>(error_before_ba) << "\n";
        if (detailedPrint) std::cout << "mean after ba: " << std::get<0>(error_after_ba) << ", rms: " << std::get<1>(error_after_ba) << "\n";

        //Only if this additional global bundle adjustment step at the very end has been run, then do we print another set of all metrics
        //TODO

        // sfm.reprojectionErrorFs << "ba," << i << "," << std::get<0>(error_after_ba) << "," << std::get<1>(error_after_ba) << "\n";

        if (sfm.writeGroundTruth)
        {
            sfm.PrintAfterFrameOutput(0.0);
        }
    }

	sfm.VisualiseSFMInitResults();
    //if (sfm.writeGroundTruth)
    //{
    //    sfm.CompareToGroundTruth();
    //}
    sfm.reprojectionErrorFs.close();
    sfm.testMetricsOutputFS.close();
    // sfm.combined_logger->flush();

	return 0;
}
