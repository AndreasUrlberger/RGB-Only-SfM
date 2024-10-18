// #include <colmap/base/reconstruction.h>
// #include <colmap/base/image_reader.h>
#include <colmap/controllers/image_reader.h>
#include <colmap/controllers/incremental_mapper.h>
#include <colmap/controllers/option_manager.h>
#include <colmap/controllers/automatic_reconstruction.h>
#include <colmap/sensor/models.h>
#include <colmap/sfm/incremental_mapper.h>
#include <colmap/util/threading.h>
#include <colmap/util/misc.h>
#include <colmap/util/logging.h> 
#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/bundle_adjustment.h"

#include <ceres/ceres.h>
#include <colmap/util/types.h>
#include <iostream>
#include <string>

// custom methods
#include "BundleAdjuster.h"
#include <spdlog/spdlog.h>

void ReconstructSubModel(colmap::IncrementalMapper& mapper,
    const colmap::IncrementalMapper::Options& mapper_options,
    const std::shared_ptr<colmap::Reconstruction>& reconstruction)
{
    // register initial pair
    // incremental mapping
    // triangulate
    // iterative local refinemnt
    // optionally run global refinemnt

}

void performColmapBundleAdjustment(const colmap::OptionManager& options, const std::shared_ptr<colmap::Reconstruction>& reconstruction)
{
  colmap::BundleAdjustmentController ba_controller(options, reconstruction);
  ba_controller.Run();
}

void performSmcBundleAdjustment(const std::shared_ptr<colmap::Reconstruction>& reconstruction)
{
    // BundleAdjuster bundleAdjuster;
    // bundleAdjuster.setupProblem();

    // std::map<size_t, double*> seen_cameras;
    // std::map<size_t, double*> seen_points;
    // std::map<size_t, size_t> camera_param_ids;
    // std::map<size_t, size_t> point_param_ids;
    // size_t i = 0;
    // for (const auto& point3D: reconstruction->Points3D())
    // {
    //     const auto& point = point3D.second;
    //     std::cout << "3D point id: " << point3D.first << std::endl;
    //     std::cout << "Number of observations: " << point.track.Length() << std::endl;
    //     std::cout << "Point: " << point3D.second.xyz(0) << "," 
    //             << point3D.second.xyz(1) << "," << point3D.second.xyz(2) << "\n";

    //     for (const auto& obs: point.track.Elements()) 
    //     {
    //         const auto image_id = obs.image_id;
    //         const auto keypoint_idx = obs.point2D_idx;

    //         const auto& image = reconstruction->Image(image_id);

    //         // get camera params
    //         auto camera_id = image.CameraId();
    //         size_t camera_param_id = 0;
    //         double* cameraParams = nullptr;
    //         if (seen_cameras.find(camera_id) == seen_cameras.end())
    //         {
    //             auto pose = image.CamFromWorld();
    //             auto rotation = pose.rotation.toRotationMatrix();
    //             auto translation = pose.translation;

    //             auto camera = reconstruction->Camera(camera_id);

    //             // Convert the rotation matrix to AngleAxis
    //             Eigen::AngleAxisd angle_axis(rotation);

    //             // Extract the rotation vector (axis * angle)
    //             Eigen::Vector3d rotation_vector = angle_axis.angle() * angle_axis.axis();

    //             // add camera param
    //             cameraParams = (double*)malloc(sizeof(double)*9);
    //             for (size_t i = 0; i < 3; ++i)
    //             {
    //                 cameraParams[i] = rotation(i);
    //                 cameraParams[i+3] = translation(i);
    //             }
    //             cameraParams[6] = camera.FocalLength();
    //             cameraParams[7] = 0;
    //             cameraParams[8] = 0;

    //             for (size_t i = 0; i < 9; ++i)
    //             {
    //                 std::cout << "camera params[" << i << "]: " << cameraParams[i] << "\n";
    //             }
    //             seen_cameras.insert({camera_id, cameraParams});
    //             camera_param_id = bundleAdjuster.addCameraParam(cameraParams);
    //             camera_param_ids.insert({camera_id, camera_param_id});
    //         }
    //         else 
    //         {
    //             cameraParams = seen_cameras.find(camera_id)->second;
    //             camera_param_id = camera_param_ids.find(camera_id)->second;
    //         }

    //         size_t point_param_id = 0;
    //         double* pointParams = nullptr;
    //         if (seen_points.find(point3D.first) == seen_points.end())
    //         {
    //             pointParams = (double*)malloc(sizeof(double)*3);
    //             pointParams[0] = point3D.second.xyz(0);
    //             pointParams[1] = point3D.second.xyz(1);
    //             pointParams[2] = point3D.second.xyz(2);
    //             for (size_t i = 0; i < 3; ++i)
    //             {
    //                 std::cout << "point params[: " << i << "]: " << pointParams[i] << "\n";
    //             }
    //             seen_points.insert({point3D.first, pointParams});
    //             point_param_id = bundleAdjuster.addWorldPointParam(pointParams);
    //             point_param_ids.insert({point3D.first, point_param_id});
    //         }
    //         else 
    //         {
    //             pointParams = seen_points.find(point3D.first)->second;
    //             point_param_id = point_param_ids.find(point3D.first)->second;
    //         }

    //         const auto& keypoint = image.Point2D(keypoint_idx);
    //         size_t observationIdx = bundleAdjuster.addObservation(
    //             camera_id,
    //             point3D.first,
    //             camera_param_id,
    //             point_param_id,
    //             keypoint.xy(0),
    //             keypoint.xy(1)
    //         );

    //         bundleAdjuster.addPointToProblem(
    //             keypoint.xy(0),
    //             keypoint.xy(1),
    //             bundleAdjuster.mutable_camera_for_observation(i),
    //             bundleAdjuster.mutable_point_for_observation(i)
    //         );
            
    //         // bundleAdjuster.addObservation(
    //         //     image.camera_
    //         // )

    //         std::cout << "Params: [" << i << "]\n";
    //         std::cout << "keypoint x: " << keypoint.xy(0) << "\n";
    //         std::cout << "keypoint y: " << keypoint.xy(1) << "\n";

    //         double* camera = bundleAdjuster.mutable_camera_for_observation(i);
    //         for (size_t j = 0; j < 9; ++j)
    //         {
    //             std::cout << "camera[" << j << "]: " << camera[j] << "\n";
    //         }

    //         double* point =  bundleAdjuster.mutable_point_for_observation(i);
    //         for (size_t j = 0; j < 3; ++j)
    //         {
    //             std::cout << "point[" << j << "]: " << point[j] << "\n";
    //         }

    //         i++;
    //     }
    // }

    // bundleAdjuster.solve();
}
int main(int argc, char** argv)
{
    // Initialize the logging
    colmap::InitializeGlog(argv);

    std::string message;
    colmap::OptionManager options;
    options.AddRequiredOption("message", &message);
    options.Parse(argc, argv);

    std::cout << colmap::StringPrintf("Hello %s!", message.c_str());
    
    // define image reader options
    // see colmap/src/pycolmap/pipeline/images.cc

    // import images from image path
    // colmap::ImageReaderOptions imageReaderOptions;
    // imageReaderOptions.database_path = "../Data/south-building/database.db";
    // imageReaderOptions.image_path = "../Data/south-building/images";
    // imageReaderOptions.image_list = {};

    // start automatic reconstruction constructor

    // 1. Initialize option manager
    options.AddAllOptions();
    *options.image_path = "../Data/south-building/images_limited_2";
    *options.database_path = "../Data/south-building/database.db";
    options.ModifyForIndividualData();
    options.sift_extraction->num_threads = 1;
    options.sift_matching->use_gpu = 0;
    options.sift_matching->num_threads = 1;
    options.sift_extraction->use_gpu = 0;

    // 2. initialize image reader
    // colmap::ImageReaderOptions imageReaderOptions;
    // imageReaderOptions.database_path = *options.database_path;
    // imageReaderOptions.image_path = *options.image_path;
    // auto feature_extractor = colmap::CreateFeatureExtractorController(imageReaderOptions,
    //     *options.sift_extraction);

    // 3. initialize exhaustive matcher
    // auto exhaustive_matcher = colmap::CreateExhaustiveFeatureMatcher(
    //     *options.exhaustive_matching,
    //     *options.sift_matching,
    //     *options.two_view_geometry,
    //     *options.database_path
    // );

    // // 4. initialize sequential matcher
    // auto sequential_matcher = colmap::CreateSequentialFeatureMatcher(
    //     *options.sequential_matching,
    //     *options.sift_matching,
    //     *options.two_view_geometry,
    //     *options.database_path
    // );

    // end automatic reconstruction constructor

    // start automatic reconstruction run()

    // 1. run feature extraction
    // if (feature_extractor != nullptr)
    // {
    //     auto active_thread = feature_extractor.get();
    //     feature_extractor->Start();
    //     feature_extractor->Wait();
    //     feature_extractor.reset();
    //     active_thread = nullptr;
    // }

    // 2. run feature matching
    // RunFeatureMatching()

    colmap::Database database(*options.database_path);
    const size_t num_images = database.NumImages();
    spdlog::info("Num images {}", num_images);
    // auto matcher = exhaustive_matcher.get();
    // if (matcher != nullptr)
    // {
    //     auto active_thread = matcher;
    //     matcher->Start();
    //     matcher->Wait();
    //     exhaustive_matcher.reset();
    //     sequential_matcher.reset();
    //     active_thread = nullptr;
    // }

    // 3. run sparse mapper
    const auto sparse_path = "../Data/south-building/sparse";
    if (colmap::ExistsDir(sparse_path))
    {
        auto dir_list = colmap::GetDirList(sparse_path);
        std::sort(dir_list.begin(), dir_list.end());
        if (dir_list.size() > 0)
        {
            spdlog::info("TODO: do something here");
        }
    }
    auto reconstruction_manager = std::make_shared<colmap::ReconstructionManager>();
    colmap::IncrementalMapperController mapper(options.mapper,
                                        *options.image_path,
                                        *options.database_path,
                                        reconstruction_manager);

    // In case a new reconstruction is started, write results of individual sub-
    // models to as their reconstruction finishes instead of writing all results
    // after all reconstructions finished.
    size_t prev_num_reconstructions = 0;
    std::string input_path = "";
    std::string output_path = "../Data/south-building/sparse";
    if (input_path == "") {
        mapper.AddCallback(
            colmap::IncrementalMapperController::LAST_IMAGE_REG_CALLBACK, [&]() {
            // If the number of reconstructions has not changed, the last model
            // was discarded for some reason.
            if (reconstruction_manager->Size() > prev_num_reconstructions) {
                const std::string reconstruction_path = colmap::JoinPaths(
                    output_path, std::to_string(prev_num_reconstructions));
                colmap::CreateDirIfNotExists(reconstruction_path);
                reconstruction_manager->Get(prev_num_reconstructions)
                    ->Write(reconstruction_path);
                options.Write(colmap::JoinPaths(reconstruction_path, "project.ini"));
                prev_num_reconstructions = reconstruction_manager->Size();
            }
            });
    }

    mapper.Run();

    // if (reconstruction_manager->Size() == 0) {
    //     LOG(ERROR) << "failed to create sparse model";
    //     return EXIT_FAILURE;
    // }

    // // In case the reconstruction is continued from an existing reconstruction, do
    // // not create sub-folders but directly write the results.
    // if (input_path != "") {
    //     const auto& reconstruction = reconstruction_manager->Get(0);

    //     // Transform the final reconstruction back to the original coordinate frame.
    //     if (options.mapper->fix_existing_images) {
    //     if (fixed_image_ids.size() < 3) {
    //         LOG(WARNING) << "Too few images to transform the reconstruction.";
    //     } else {
    //         std::vector<Eigen::Vector3d> new_fixed_image_positions;
    //         new_fixed_image_positions.reserve(fixed_image_ids.size());
    //         for (const image_t image_id : fixed_image_ids) {
    //         new_fixed_image_positions.push_back(
    //             reconstruction->Image(image_id).ProjectionCenter());
    //         }
    //         Sim3d orig_from_new;
    //         if (EstimateSim3d(new_fixed_image_positions,
    //                         orig_fixed_image_positions,
    //                         orig_from_new)) {
    //         reconstruction->Transform(orig_from_new);
    //         } else {
    //         LOG(WARNING) << "Failed to transform the reconstruction back "
    //                         "to the input coordinate frame.";
    //         }
    //     }
    //     }

    //     reconstruction->Write(output_path);
    // }
    // colmap::AutomaticReconstructionController::Options reconstruction_options;
    // reconstruction_options.workspace_path = "../Data/south-building";
    // reconstruction_options.image_path = *options.image_path;
    // reconstruction_options.use_gpu = 0;
    // colmap::AutomaticReconstructionController controller(
    //     reconstruction_options,
    //     reconstruction_manager
    // );

    // controller.Start();
    // controller.Wait();
    // // colmap::IncrementalMapperController mapper(options.mapper,
    // //     *options.image_path,
    // //     *options.database_path,
    // //     reconstruction_manager);
    // // mapper.SetCheckIfStoppedFunc([&]() { return true; });
    // // mapper.Run();
    // // start sparse mapper run()
    // {
    //     colmap::Timer run_timer;
    //     run_timer.Start();
        
    //     // start load database
    //     spdlog::info("Loading database");
    //     colmap::Database database(*options.database_path);
    //     colmap::Timer timer;
    //     timer.Start();
    //     const size_t min_num_matches = 15; // TODO: fix?
    //     auto database_cache = colmap::DatabaseCache::Create(
    //         database, min_num_matches,
    //         true, 
    //         {}
    //     );
    //     timer.PrintMinutes();

    //     if (database_cache->NumImages() == 0) 
    //     {
    //         spdlog::warn("No images with matches found in the database");
    //     }
    //     else 
    //     {
    //         spdlog::info("Num images in database with matches {}", database_cache->NumImages());
    //     }

    //     // end load database

    //     // start reconstruct
    //     // colmap::IncrementalMapper::Options init_mapper_options;
    //     // colmap::IncrementalMapper mapper(&database, reconstruction_options.image_path, init_mapper_options);


    //     const bool initial_reconstruction_given = reconstruction_manager->Size() > 0;
    //     auto CheckIfStopped = []() {
    //         return false;
    //     };
    //     int init_num_trials = 1;
    //     for (int num_trials = 0; num_trials < init_num_trials; ++num_trials)
    //     {
    //         if (CheckIfStopped()) {
    //             break;
    //         }
    //         size_t reconstruction_idx;
    //         if (!initial_reconstruction_given || num_trials > 0)
    //         {
    //             reconstruction_idx = reconstruction_manager->Add();
    //         }
    //         else 
    //         {
    //             reconstruction_idx = 0;
    //         }

    //         std::shared_ptr<colmap::Reconstruction> reconstruction =
    //             reconstruction_manager->Get(reconstruction_idx);

    //         std::cout << "cameras: " << reconstruction->Cameras().size() << "\n";
            
    //         // camera params
    //         for (const auto& camera: reconstruction->Cameras())
    //         {
    //             std::cout << "camera: " << camera.first << " intrinsic params: " 
    //                 << camera.second.ParamsToString() << std::endl;
    //         }

    //         // image poses
    //         for (const auto& image: reconstruction->Images())
    //         {
    //             auto& cam = image.second.CamFromWorld();
    //             std::cout << "Image " << image.second.Name() << " pose: " 
    //                 << cam.rotation << "\n" << cam.translation << "\n";
    //         }

    //         // 3d points
    //         size_t num_observations  = 0;
    //         for (const auto& point3D: reconstruction->Points3D())
    //         {
    //             const Eigen::Vector3d& xyz = point3D.second.xyz;
    //             std::cout << "3D Point: " << point3D.first << " coordinates: "
    //                 << xyz << ","
    //                 << " num of observations: " << point3D.second.track.Length() << "\n"; 
    //             num_observations += point3D.second.track.Length();
    //         }

    //         std::cout << "total num observations: " << num_observations << "\n";

    //         std::cout << "Num points: " << reconstruction->NumPoints3D() << ","
    //             << "num poses: " << reconstruction->NumImages() << std::endl;

    //         auto error = reconstruction->ComputeMeanReprojectionError();

    //         // performSmcBundleAdjustment(reconstruction);
    //         performColmapBundleAdjustment(options, reconstruction);

    //         std::cout << "mean reprojection error before: " << error << "\n";
    //         std::cout << "mean reprojection error after: " << reconstruction->ComputeMeanReprojectionError() << std::endl;
    //         // see sfm.cc

    //         // const colmap::IncrementalMapperController::Status status = ReconstructSubModel(mapper, mapper_options, reconstruction);
    //         // for (const auto& point3D: reconstruction->Points3D())
    //         // {
    //         //     const Eigen::Vector3d& xyz = point3D.second.xyz;
    //         //     std::cout << "3D Point after adjustment: " << point3D.first << " coordinates: "
    //         //         << xyz << ","
    //         //         << " num of observations: " << point3D.second.track.Length() << "\n"; 
    //         // }
    //     }
    //     // end reconstruct
    // }
    // // optionally write reconstruction

    colmap::CreateDirIfNotExists(sparse_path);
    reconstruction_manager->Write(sparse_path);
    options.Write(colmap::JoinPaths(sparse_path, "project.ini"));
    
    // end automatic reconstruction run()

    // BundleAdjuster bundleAdjuster;
    // bundleAdjuster.setupProblem();
    // bundleAdjuster.solve();

    std::cout << "\n";
    // std::cout << "Read " << num_iterations << " images\n";

    return EXIT_SUCCESS;

    // // Define options for image reader
    // colmap::ImageReaderOptions image_reader_options;
    // image_reader_options.database_path = "../Data/south-building/database.db";
    // image_reader_options.image_path = "../Data/south-building/images";
    // colmap::Database db;
    // // image_reader_options.camera_model = colmap::SimplePinholeCameraModel::model_id;
    // image_reader_options.single_camera = true;

    // // Initialize image reader
    // colmap::ImageReader image_reader(image_reader_options, &db);

    // // Define options for incremental mapper
    // colmap::Database database(image_reader_options.database_path);
    // auto database_cache = colmap::DatabaseCache::Create(database, 0, false, {});

    // colmap::IncrementalMapper::Options mapper_options;
    // colmap::IncrementalMapper mapper(database_cache);

    // // Initialize a new reconstruction
    // colmap::Reconstruction reconstruction;
    // mapper.BeginReconstruction(std::make_shared<colmap::Reconstruction>(reconstruction));
    // while (image_reader.NextIndex() < image_reader.NumImages())
    // {
    //     colmap::Camera camera;
    //     colmap::Image image;
    //     colmap::Bitmap bitmap;
    //     colmap::Bitmap mask;

    //     image_reader.Next(&camera, &image, &bitmap, &mask);
    //     auto image_id = image.ImageId();

    //     mapper.RegisterNextImage(mapper_options, image_id);
    // }

    // ceres::Problem problem;

    // std::cout << "Num cameras: " << reconstruction.NumCameras() << "\n";


    return 0;
}