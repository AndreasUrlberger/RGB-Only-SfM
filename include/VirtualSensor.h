#pragma once

#include <cassert>
#include <vector>
#include <iostream>
#include <cstring>
//#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "Eigen.h"

typedef unsigned char BYTE;

enum ImageType
{
	Kinect, Colmap, Colmap_640_480
};

// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor
{
public:

	VirtualSensor() : m_currentIdx(-1), m_increment(1)
	{

	}

	~VirtualSensor()
	{
		std::cout << "Destructor called\n";
		SAFE_DELETE_ARRAY(m_depthFrame);
		SAFE_DELETE_ARRAY(m_colorFrame);
	}

	bool Init(const std::string& datasetDir, ImageType typeOfImages, bool hasDepth, bool hasTrajectory)
	{
		m_imageType = typeOfImages;
		m_hasDepth = hasDepth;
		m_hasTrajectory = hasTrajectory;
		m_baseDir = datasetDir;
        m_scale = 3.47495;
        
		// read filename lists
		bool readRGBSuccess;
		if (m_hasDepth)
		{
			if (!ReadFileList(datasetDir + "depth.txt", m_filenameDepthImages, m_depthImageIDs)) 
			{
				std::cout << "Depth can't be read \n";
				return false;
			}
			else
			{
				std::cout << "Depth has been read successfully \n";
			}
		}

		if (!ReadFileList(datasetDir + "rgb.txt", m_filenameColorImages, m_colorImageIDs)) 
		{
			std::cout << "RGB can't be read \n";
			return false;
		}
		else 
		{
			std::cout << "RGB has been read successfully \n";
		}

		// read tracking
		if (m_hasTrajectory)
		{
			if (!ReadTrajectoryFile(datasetDir + "groundtruth.txt", m_trajectory, m_trajectoryIDs))
			{
				std::cout << "Tracking can't be read \n";
				return false;
			} 
			else
			{
				std::cout << "Tracking has been read successfully \n";
			}
		}

		if (m_hasDepth && (m_filenameDepthImages.size() != m_filenameColorImages.size())) 
		{
			std::cout << "Error: Number of depth images does not match number of color images! \n";
			return false;
		}

		switch (m_imageType)
		{
			case Kinect:
				m_colorImageWidth = 640;
				m_colorImageHeight = 480;
				
				//TUM RGB-D Dataset
				m_colorIntrinsics <<
					517.3f, 0.0f, 318.6f, 
					0.0f, 516.5f, 255.3f,
					0.0f, 0.0f, 1.0f;

				break;

			case Colmap:
				m_colorImageWidth = 3072;
				m_colorImageHeight = 2304;

				//south building Dataset
				m_colorIntrinsics <<
					2559.68f, 0.0f, 1536.0f,
					0.0f, 2559.68f, 1152.0f,
					0.0f, 0.0f, 1.0f;

				break;
			
			case Colmap_640_480:
				m_colorImageWidth = 640;
				m_colorImageHeight = 480;

				// Adjusted for the 4.8 times smaller images.
				m_colorIntrinsics <<
					2559.68f/4.8f, 0.0f,          1536.0f/4.8f, 
					0.0f,          2559.68f/4.8f, 1152.0f/4.8f,
					0.0f,          0.0f,          1.0f;
				
				break;

			default:
				std::cerr << "No Image type given!" << std::endl;
				break;
		}

		m_depthImageWidth = m_colorImageWidth;
		m_depthImageHeight = m_colorImageHeight;
		m_depthIntrinsics = m_colorIntrinsics;

		m_colorExtrinsics.setIdentity();
		m_depthExtrinsics.setIdentity();

		m_depthFrame = new float[m_depthImageWidth*m_depthImageHeight];
		for (unsigned int i = 0; i < m_depthImageWidth*m_depthImageHeight; ++i)
		{
			m_depthFrame[i] = 0.5f;
		}

		m_colorFrame = new BYTE[4* m_colorImageWidth*m_colorImageHeight];
		for (unsigned int i = 0; i < 4*m_colorImageWidth*m_colorImageHeight; ++i)
		{
			m_colorFrame[i] = 255;
		} 

		m_currentIdx = -1;

		return true;
	}

	bool ProcessNextFrame()
	{
		if (m_currentIdx == -1)	m_currentIdx = 0;
		else m_currentIdx += m_increment;
		//m_scale = 2.0;
		
		if ((unsigned int)m_currentIdx >= (unsigned int)m_filenameColorImages.size()) return false;

		std::cout << "ProcessNextFrame [" << m_currentIdx << " | " << m_filenameColorImages.size() << "]-----------------------------------------------" << std::endl;

		//img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
		/*for (int i = 0; i < 15; i++) {
			std::cout << m_filenameColorImages[i] << "\n";
		}*/
		//std::cout << "Trying to read image " << m_baseDir + m_filenameColorImages[m_currentIdx] << "\n";
		cv::Mat rgbImage = cv::imread(m_baseDir + m_filenameColorImages[m_currentIdx], cv::IMREAD_UNCHANGED);
		cv::cvtColor(rgbImage, rgbImage, cv::COLOR_BGR2BGRA);

		if (rgbImage.empty()) {
			std::cerr << "Error: Could not load the image: " << m_baseDir + m_filenameColorImages[m_currentIdx] << std::endl;
			return false;
		}
		memcpy(m_colorFrame, rgbImage.data, 4 * m_colorImageWidth * m_colorImageHeight);

		// Load the current depth image using OpenCV
		if (m_hasDepth)
		{
			cv::Mat dImage = cv::imread(m_baseDir + m_filenameDepthImages[m_currentIdx], cv::IMREAD_ANYDEPTH);
			if (dImage.empty()) {
				std::cerr << "Error: Could not load the image: " << m_baseDir + m_filenameDepthImages[m_currentIdx] << std::endl;
				return false;
			}

			//// depth images are scaled by 5000 (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
			//FreeImageU16F dImage; //Here we get the current depth image
			//dImage.LoadImageFromFile(m_baseDir + m_filenameDepthImages[m_currentIdx]);

			//NOTE: when we do our bundle adjustment with an additional energy term for the depth, then we need to access the depth not through the opencv depth image, but instead through the float array "m_depthFrame" as this is scaled correctly (done below)

			for (unsigned int i = 0; i < m_depthImageWidth*m_depthImageHeight; ++i)
			{
				if (dImage.data[i] == 0)
					m_depthFrame[i] = MINF;
				else
					m_depthFrame[i] = dImage.data[i] * 1.0f / 5000.0f;
			}

			
		}
		if (m_hasTrajectory)
		{
			if (m_imageType == Kinect)
			{
				// find pose transformation (simple nearest neighbor, linear search) in list of trajectories
				double image_id = m_colorImageIDs[m_currentIdx];
			
				double min = std::numeric_limits<double>::max();
				int idx = 0;
				for (unsigned int i = 0; i < m_trajectory.size(); ++i)
				{
					double d = fabs(m_trajectoryIDs[i] - image_id); //Find the closest image id matching the current one
					if (min > d)
					{
						min = d;
						idx = i;
					}
				}
				m_currentTrajectory = m_trajectory[idx];
				//std::cout << "In processNextFrame(). m_currentIdx is " << m_currentIdx << ", and current image_id is " << image_id << ", closest matching id found in list of trajectories is " << m_trajectoryIDs[idx] << "\n";
			}
			else if (m_imageType == Colmap || m_imageType == Colmap_640_480)
			{
				//Here, there are no timestamps, but just the names of the images
				//Thus, we dont have to find the nearest, we can just find the exact match
				int image_id = m_colorImageIDs[m_currentIdx]; 

				int idx = 0;
				for (unsigned int i = 0; i < m_trajectory.size(); ++i)
				{
					if (m_trajectoryIDs[i] == image_id)
					{
						idx = i;
						break;
					}
				}
				m_currentTrajectory = m_trajectory[idx];
				//std::cout << "In processNextFrame(). m_currentIdx is " << m_currentIdx << ", and current image_id is " << image_id << ", closest matching id found in list of trajectories is " << m_trajectoryIDs[idx] << "\n";

			}
			
		}

		return true;
	}

	unsigned int GetCurrentFrameCnt()
	{
		return (unsigned int)m_currentIdx;
	}

	// get current color data
	BYTE* GetColorRGBX()
	{
		return m_colorFrame;
	}

	// get current depth data
	float* GetDepth()
	{
		return m_depthFrame;
	}

	// color camera info
	Eigen::Matrix3f GetColorIntrinsics()
	{
		return m_colorIntrinsics;
	}

	// color camera info
	cv::Mat GetColorIntrinsics(int i)
	{
		double fx = m_colorIntrinsics(0, 0);
        double cx = m_colorIntrinsics(0, 2);
        double cy = m_colorIntrinsics(1, 2);

		cv::Mat temp = cv::Mat::eye(3, 3, CV_32F);
		temp.at<float>(0, 0) = fx;
		temp.at<float>(1, 1) = fx;
		temp.at<float>(0, 2) = cx;
		temp.at<float>(1, 2) = cy;

		return temp;
	}

	Eigen::Matrix4f GetColorExtrinsics()
	{
		return m_colorExtrinsics;
	}

	unsigned int GetColorImageWidth()
	{
		return m_colorImageWidth;
	}

	unsigned int GetColorImageHeight()
	{
		return m_colorImageHeight;
	}

	// depth (ir) camera info
	Eigen::Matrix3f GetDepthIntrinsics()
	{
		return m_depthIntrinsics;
	}

	Eigen::Matrix4f GetDepthExtrinsics()
	{
		return m_depthExtrinsics;
	}

	unsigned int GetDepthImageWidth()
	{
		return m_colorImageWidth;
	}

	unsigned int GetDepthImageHeight()
	{
		return m_colorImageHeight;
	}

	// get current trajectory transformation
	Eigen::Matrix4f GetTrajectory()
	{
		return m_currentTrajectory;
	}

	double GetScale()
	{ 
		return m_scale;
	}

	void SetScale(double newScale)
	{ 
		m_scale = newScale;
	}

	void SetIncrement(int newIncrement)
	{
		assert(newIncrement > 0 && "Increment must be greater than 0");
		m_increment = newIncrement;
	}

private:

	bool ReadFileList(const std::string& filename, std::vector<std::string>& result, std::vector<double>& image_ids)
	{
		std::ifstream fileDepthList(filename, std::ios::in);
		if (!fileDepthList.is_open()) return false;
		result.clear();
		image_ids.clear();
		std::string dump;
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		while (fileDepthList.good())
		{
			//For TUM dataset, the image_id is the timestamp of the image, for south building, it is the name of the image
			double image_id;
			if (m_imageType == Kinect)
			{
				//This is for the TUM dataset
				//Here the image_id is its timestamp
				fileDepthList >> image_id; 
			}
			
			std::string filename;
			fileDepthList >> filename;
			if (filename == "") break;

			if (m_imageType == Colmap || m_imageType == Colmap_640_480)
			{
				std::size_t found = filename.find_last_of("/P");
				std::size_t found2 = filename.substr(found + 1).find_last_of(".");
				//std::string image_id = 
				image_id = std::stod(filename.substr(found + 1).substr(0, found2));
				//std::cout << "In read file list, the image_id of the colmap image being read in is " << (int) image_id << "\n";
			}

			image_ids.push_back(image_id);
			result.push_back(filename);
		}
		fileDepthList.close();
		return true;
	}

	bool ReadTrajectoryFile(const std::string& filename, std::vector<Eigen::Matrix4f>& result, std::vector<double>& trajectoryIDs)
	{
		int counter = 0;
		bool printDebug = false;
		if (m_imageType == Kinect)
		{
			std::cout << "Reading trajectory with image type Kinect \n";
			std::ifstream file(filename, std::ios::in);
			if (!file.is_open()) return false;
			result.clear();
			std::string dump;
			std::getline(file, dump);
			std::getline(file, dump);
			std::getline(file, dump);

			while (file.good())
			{
				double timestamp;
				file >> timestamp;
				Eigen::Vector3f translation;
				file >> translation.x() >> translation.y() >> translation.z();
				double qw, qx, qy, qz;
				file >> qx >> qy >> qz >> qw;
				
				Eigen::Quaternionf rot(qw, qx, qy, qz);
				//Eigen::Quaternionf rot;
				//file >> rot;

				Eigen::Matrix4f transf;
				transf.setIdentity();
				transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
				transf.block<3, 1>(0, 3) = translation;

				if (printDebug && counter < 25) 
				{
					std::cout << "-----------\n";
					std::cout << "counter is " << counter << "\n";
 					std::cout << "qw, qx, qy, qz are " << qw << ", " << qx << ", " << qy << ", " << qz << "\n";
				
					Eigen::Quaternionf r_truth(rot.toRotationMatrix());
					std::cout << "Ground truth rot mat is " << rot.toRotationMatrix() << "\n";
					std::cout << "Ground truth rot quat is " << r_truth << "\n";

					std::cout << "-----------\n";
					counter++;
				}
				

				if (rot.norm() == 0) break;

				transf = transf.inverse().eval();

				trajectoryIDs.push_back(timestamp);
				result.push_back(transf);
			}
			file.close();
			return true;
		}
		else if (m_imageType == Colmap || m_imageType == Colmap_640_480)
		{
			std::cout << "Reading trajectory with image type either colmap of colmap_640_480 \n";

			std::ifstream file(filename, std::ios::in);
			if (!file.is_open()) return false;
			result.clear();
			std::string dump;
			std::getline(file, dump);
			std::getline(file, dump);
			std::getline(file, dump);
			std::getline(file, dump);

			std::map<int, Eigen::Matrix4f> transfsMap;

			int counter = 0;

			do 
			{
				if (counter >= m_colorImageIDs.size())
				{
					break;
				}
				

				counter++;
				int temp_image_id;
				file >> temp_image_id;

				
				
				double qw, qx, qy, qz;
				file >> qw >> qx >> qy >> qz;
				
				Eigen::Quaternionf rot(qw, qx, qy, qz);

				

				Eigen::Vector3f translation;
				file >> translation.x() >> translation.y() >> translation.z();
				

				int camId;
				file >> camId;

				std::string imageName;
				file >> imageName;

				double image_id;
				std::size_t found = imageName.find_last_of(".");
				std::size_t found2 = imageName.substr(0, found).find_last_of("P");
				image_id = std::stod(imageName.substr(0, found).substr(found2 + 1));

				

				Eigen::Matrix4f transf;
				transf.setIdentity();
				transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
				transf.block<3, 1>(0, 3) = translation;

				Eigen::Matrix3f eigenRMat = transf.block<3, 3>(0, 0);

				Eigen::Quaternionf rot2(eigenRMat);
				

				if (printDebug)
				{
					std::cout << "Temp_image_id is " << temp_image_id << ", ";
					std::cout << "qw, qx, qy, qz are " << qw << ", " << qx << ", " << qy << ", " << qz << ", \n";
					std::cout << "Quaternion created from the above coeffs looks like " << rot << "\n";
					std::cout << "translation is " << translation;
					std::cout << "Image_id is " << (int) image_id << "\n";
					std::cout << "Quaternion created from the rotMat created from the above quat is " << rot2 << "\n";
					std::cout << "Counter is " << counter << "\n";
					std::cout << "---\n";	
				}

				if (rot.norm() == 0) break;

				//We do not need to do this for the colmap datasets. I suspect it is because their given camera poses are already world space positions.
				//transf = transf.inverse().eval(); //BJORN_QUESTION: Why do we do this? TO then just later invert the transformation again?

				trajectoryIDs.push_back(image_id);
				result.push_back(transf);

				std::getline(file, dump);

				

			} while (std::getline(file, dump));

			file.close();
			return true;
		}

		return false;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// current frame index
	int m_currentIdx;

	int m_increment;

	// frame data
	float* m_depthFrame;
	BYTE* m_colorFrame;
	bool m_hasTrajectory;
	Eigen::Matrix4f m_currentTrajectory;

	// color camera info
	Eigen::Matrix3f m_colorIntrinsics;
	Eigen::Matrix4f m_colorExtrinsics;
	unsigned int m_colorImageWidth;
	unsigned int m_colorImageHeight;

	// depth (ir) camera info
	bool m_hasDepth;
	Eigen::Matrix3f m_depthIntrinsics;
	Eigen::Matrix4f m_depthExtrinsics;
	unsigned int m_depthImageWidth;
	unsigned int m_depthImageHeight;

	// base dir
	std::string m_baseDir;
	// filenamelist depth
	std::vector<std::string> m_filenameDepthImages;
	std::vector<double> m_depthImageIDs;
	// filenamelist color
	std::vector<std::string> m_filenameColorImages;
	std::vector<double> m_colorImageIDs;

	// trajectory
	std::vector<Eigen::Matrix4f> m_trajectory;
	std::vector<double> m_trajectoryIDs;

	//scale
    double m_scale;

	//Image resolution
	ImageType m_imageType = Kinect;
};