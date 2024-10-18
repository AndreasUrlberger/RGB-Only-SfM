#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen.h"

// Define a struct to hold the camera pose data
struct CameraPose 
{
    double timestamp;
    Vector3f translation;
    Matrix3f rotation;
};

// Function to parse a line and return a CameraPose struct
CameraPose parseLine(const std::string& line) 
{
    std::istringstream iss(line);
    CameraPose pose;
    float tx, ty, tz, qx, qy, qz, qw;
    
    iss >> pose.timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw ;
    
    pose.translation = Vector3f(tx, ty, tz);

    // Convert quaternion to rotation matrix
    Quaternionf quaternion(qw, qx, qy, qz);
    Matrix3f rotationMatrix = quaternion.toRotationMatrix();

    //// Convert Eigen::Matrix3f to cv::Mat
    //cv::Mat cvRotationMatrix(3, 3, CV_32F);
    //for (int i = 0; i < 3; ++i) {
    //    for (int j = 0; j < 3; ++j) {
    //        cvRotationMatrix.at<float>(i, j) = rotationMatrix(i, j);
    //    }
    //}

    pose.rotation = rotationMatrix;
    return pose;
}

//Function to read camera poses from a file and store them in a vector
std::vector<CameraPose> readCameraPoses(const std::string& filename) 
{
    std::ifstream file(filename);
    std::vector<CameraPose> poses;
    std::string line;

    // Read file line by line
    while (std::getline(file, line)) 
    {
        if (!line.empty() && line.at(0) != '#' )     // Skip empty lines and comments
        { 
            //CameraPose pose = parseLine(line);
            CameraPose pose = CameraPose();
            poses.push_back(pose);
        }
    }

    return poses;
}

void printFileRead(std::string filePath)
{
    //std::vector<CameraPose> cameraPoses = readCameraPoses(filePath);
    std::vector<CameraPose> cameraPoses;
    
    // Output the parsed camera poses
    for (const auto& pose : cameraPoses) 
    {
        std::cout << "Timestamp: " << pose.timestamp
            << ", Translation: (" << pose.translation.x() << ", " << pose.translation.y() << ", " << pose.translation.z() << ")"
            //<< ", Rotation: (" << pose.rotation.x() << ", " << pose.rotation.y() << ", " << pose.rotation.z() << ", " 
            //<< pose.rotation.w() << ")"
            << ", Rotation: " << pose.rotation
            << std::endl;
    }

}