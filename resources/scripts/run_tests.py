import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define the default values for the config file
default_config = {
    "dataset_path": "../Data/rgbd_dataset_freiburg1_xyz/",
    "run_headless": True,
    "write_ground_truth_camera_poses": True,
    "ground_truth_poses_scale_type": 2, 
    "ground_truth_poses_scale_factor": -1.0, 
    "feature_match_ratio_threshold": 0.70, 
    "frame_interval": 10,
    "dataset_has_depth": True,
    "dataset_has_ground_truth_trajectory": True,
    "camera_type": "kinect", 
    "num_frames": 80,
    "blurry_threshold": 1.1,
    "num_features_to_detect": -1,
    "min_reprojection_error_threshold": 2.0, 
    "reprojection_error_max_std_devs": 2.0,
    "run_local_bundle_adjustment": False,
    "run_global_bundle_adjustment": False,
    "filter_if_not_seen": True,
    "filter_if_not_seen_n_times": 5,
    "filter_if_not_seen_n_times_in_m_frames": 5,
    "loss_function_type": "CAUCHY" # Possible values: "TRIVIAL", "CAUCHY", "HUBER"
}

# Define the wanted values for each parameter
wanted = {
    "dataset": [
        # {
        #     "dataset_path": "../Data/south_building_small/", 
        #     "dataset_has_depth": False,
        #     "dataset_has_ground_truth_trajectory": True,
        #     "camera_type": "colmap_640_480",
        #     "frame_interval": 1,
        # },
        {
            "dataset_path": "../Data/rgbd_dataset_freiburg1_xyz/", 
            "dataset_has_depth": True,
            "dataset_has_ground_truth_trajectory": True,
            "camera_type": "kinect",
            "frame_interval": 10,
        },
    ],
    "feature_match_ratio_threshold": [0.70],#[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    "bundle_adjustment": [
        {
            "run_global_bundle_adjustment": True,
            "run_local_bundle_adjustment": False,  
        },
        # {
        #     "run_global_bundle_adjustment": False,
        #     "run_local_bundle_adjustment": True,  
        # },
        {
            "run_global_bundle_adjustment": False,
            "run_local_bundle_adjustment": False,  
        },
    ],
    "filter_if_not_seen": [
        # {
        #     "filter_if_not_seen": False,
        # },
        # {
        #     "filter_if_not_seen": True,
        #     "filter_if_not_seen_n_times": 3,
        #     "filter_if_not_seen_n_times_in_m_frames": 3
        # },
        # {
        #     "filter_if_not_seen": True,
        #     "filter_if_not_seen_n_times": 3,
        #     "filter_if_not_seen_n_times_in_m_frames": 4
        # },
        # {
        #     "filter_if_not_seen": True,
        #     "filter_if_not_seen_n_times": 4,
        #     "filter_if_not_seen_n_times_in_m_frames": 4
        # },
        {
            "filter_if_not_seen": True,
            "filter_if_not_seen_n_times": 4,
            "filter_if_not_seen_n_times_in_m_frames": 5
        },
        # {
        #     "filter_if_not_seen": True,
        #     "filter_if_not_seen_n_times": 5,
        #     "filter_if_not_seen_n_times_in_m_frames": 5
        # }
    ],
    "loss_function_type": ["CAUCHY"],#["TRIVIAL", "CAUCHY", "HUBER"],
    "blurry_threshold": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0],
}

num_of_cores = 9
current_dir = os.getcwd()
print("current working directory:", current_dir)
print("Please run this script from the main project directory.")

config_output_dir = os.path.join(current_dir, "resources/config_automatic_tests/")
executable_path = os.path.join(current_dir, "build/3dsmc_project")
print("config_output_dir:", config_output_dir)
print("executable_path:", executable_path)

# Ensure the config directory exists
os.makedirs(config_output_dir, exist_ok=True)

# Function to execute with ProcessPoolExecutor
def run_sfm(config, config_file_path):    
    # Set the environment variable to the config file path
    os.environ["CONFIG_PATH"] = config_file_path

    # Execute the SFM process
    success = os.system(executable_path)
    
    if success != 0:
        return f"SFM did not exit cleanly for config: {config_file_path}"
    return f"SFM completed successfully for config: {config_file_path}"

# Generate configurations
configs = []
for dataset in wanted["dataset"]:
    for loss_function_type in wanted["loss_function_type"]:
        for bundle_adjustment in wanted["bundle_adjustment"]:
            for filter_if_not_seen in wanted["filter_if_not_seen"]:
                for feature_match_ratio_threshold in wanted["feature_match_ratio_threshold"]:
                    for blurry_threshold in wanted["blurry_threshold"]:
                        # Create json config file (dictionary)
                        config = default_config.copy()
                        config.update(dataset)
                        config.update(bundle_adjustment)
                        config.update(filter_if_not_seen)
                        config["feature_match_ratio_threshold"] = feature_match_ratio_threshold
                        config["loss_function_type"] = loss_function_type
                        config["blurry_threshold"] = blurry_threshold

                        config_file_name = f"config_{len(configs)}.json"
                        config_file_path = os.path.join(config_output_dir, config_file_name)
                        configs.append((config, config_file_path))

                        # Write the config to the file
                        with open(config_file_path, "w") as f:
                            json.dump(config, f)

# Change to the build directory
os.chdir(os.path.join(current_dir, "build"))

# Run configurations in parallel
with ProcessPoolExecutor(max_workers=num_of_cores) as executor:
    futures = [executor.submit(run_sfm, config, config_file_path) for config, config_file_path in configs]
    for future in as_completed(futures):
        print(future.result())

# Clean up
for _, config_file_path in configs:
    os.remove(config_file_path)
os.environ["CONFIG_PATH"] = ""