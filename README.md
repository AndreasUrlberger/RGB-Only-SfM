# 3dsmc-project

3D Scanning and Motion Capture project, implementing structure-from-motion for RGB and RGB-D inputs.

## Dependencies

* Eigen 
* Ceres
* FreeImage
* OpenCV 4

Use the provided Docker image in ```dockerfiles/Dockerfile```.

## GUI Support in Docker

### MacOS

1. Install (xQuartz)[https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088] to enable X11 forwarding
2. Ensure you are using latest Docker image (requires GTK and OpenCV)
3. Ensure `run_headless` set to `false` in OpenCV config

## DevContainer (local development)

TODO:

## Github Codespaces (remote development)

TODO:

## Github CI/CD

Currently, CI/CD using the Github actions workflow defined in `.github/workflows/main.yml` will run for all branches. At the moment, it simply clones the repository, configures using CMake, builds, and then runs. It runs on an `ubuntu-runner` using the `ghcr.io/3dsmc/3dsmc-project:latest` image.


## Instructions for use
Edit the config.json file in resources/config/ to configure the properties for how to run the SfM pipeline. Currently, this pipeline only works with two datasets:

* The TUM-RGB Dataset "freiburg1_xyz" https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
* The Colmap Dataset "South_Building". https://colmap.github.io/datasets.html To save on processing time, we used the downscaled version of the images, which are not included, as they are too large to upload on Moodle. If you choose to downscale the images from the dataset, scale them to 640x480.

When changing the path to the dataset images, make sure to also change the type of image used. OPtions are: "kinect", "colmap", and "colmap_640_480". And also provide the ground truth files from the resepective datasets, or disable this option in the config.json file.
