import glob
import re
import math
from pathlib import Path
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def get_hyperparameters(filename):
    """
    Hyperparameters are stored in the filename. Need to use regex to extract them.
    """
    hyperparameters = {}
    # Extract the hyperparameters from the filename
    # Filename = Metrics_Output_{num_frames}f_{match_threshold}mth_{frame_interval}fi_{run_local_ba}_{run_global_ba}_{filter_if_not_seen}_{filter_if_not_seen_n_times}_{filter_if_not_seen_n_times_in_m_frames}_{min_reprojection_error_threshold}mret_{reprojection_error_max_std_devs}remsd_{blurry_image_threshold}_{loss_function_type}
    # Example: Metrics_Output_50f_0.70mth_1fi_nolocalba_globalba_fins_3_in_4_2.00mret_2.00remsd_1.1_CAUCHY
    match = re.match(r"Metrics_Output_(\d+)f_(\d+\.\d+)mth_(\d+)fi_(no)?localba_(no)?globalba_((nofins)|(fins_(\d+)_in_(\d+)))_(\d+\.\d+)mret_(\d+\.\d+)remsd(_(\d+\.\d+))?_(\w+)", filename)

    if not match:
        raise ValueError(f"Filename {filename} does not match the expected format.")
    
    # Read front to end, parse values, then cut off that part of the string
    # Number of frames
    pattern = "(?:Metrics_Output_)(\d+)(?:f)"
    hyperparameters["num_frames"] = int(re.match(pattern, filename).group(1))
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Match threshold
    pattern = "(?:_)(\d+\.\d+)(?:mth)"
    hyperparameters["match_threshold"] = float(re.match(pattern, filename).group(1))
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Frame interval
    pattern = "(?:_)(\d+)(?:fi)"
    hyperparameters["frame_interval"] = int(re.match(pattern, filename).group(1))
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Run local BA
    pattern = "(?:_)((?:no)?localba)"
    hyperparameters["run_local_ba"] = not re.match(pattern, filename).group(1).startswith("no")
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Run global BA
    pattern = "(?:_)((?:no)?globalba)"
    hyperparameters["run_global_ba"] = not re.match(pattern, filename).group(1).startswith("no")
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Filter if not seen
    pattern = "(?:_)((?:nofins)|(?:fins_(\d)+_in_(\d+)))"
    if re.match(pattern, filename).group(1).startswith("no"):
        hyperparameters["filter_if_not_seen"] = False
        hyperparameters["filter_if_not_seen_n_times"] = 0
        hyperparameters["filter_if_not_seen_n_times_in_m_frames"] = 0
    else:
        hyperparameters["filter_if_not_seen"] = True
        hyperparameters["filter_if_not_seen_n_times"] = int(re.match(pattern, filename).group(2))
        hyperparameters["filter_if_not_seen_n_times_in_m_frames"] = int(re.match(pattern, filename).group(3))
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Min reprojection error threshold
    pattern = "(?:_)(\d+\.\d+)(?:mret)"
    hyperparameters["min_reprojection_error_threshold"] = float(re.match(pattern, filename).group(1))
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Reprojection error max std devs
    pattern = "(?:_)(\d+\.\d+)(?:remsd)"
    hyperparameters["reprojection_error_max_std_devs"] = float(re.match(pattern, filename).group(1))
    filename = filename[len(re.match(pattern, filename).group(0)):]

    # Blurry image threshold
    pattern = "(?:_)(\d+\.\d+)"
    if re.match(pattern, filename):
        hyperparameters["blurry_image_threshold"] = float(re.match(pattern, filename).group(1))
        filename = filename[len(re.match(pattern, filename).group(0)):]
    else:
        hyperparameters["blurry_image_threshold"] = 1.1

    # Loss function type
    pattern = "(?:_)(\w+)"
    hyperparameters["loss_function_type"] = re.match(pattern, filename).group(1)
    filename = filename[len(re.match(pattern, filename).group(0)):]

    return hyperparameters
    
def get_final_result_data(list_of_files):
    data = []
    for file in list_of_files:
        sample = {}
        title = Path(file).stem
        hyperparameters = get_hyperparameters(title)
        sample["hyperparameters"] = hyperparameters
        sample["filename"] = file

        with open(file) as csvfile:
            metrics_reader = csv.reader(csvfile, delimiter=',')
            header = next(metrics_reader, None)
            if header is None:
                print(f"File {file} is empty.")
                continue
        
            header = [h.strip() for h in header]

            index_mean_error_t_x = header.index('cam_mean_t_x_err')
            index_mean_error_t_y = header.index('cam_mean_t_y_err')
            index_mean_error_t_z = header.index('cam_mean_t_z_err')
            index_time = header.index('time')
            index_total_time = header.index('total_time')
            index_n_world_points = header.index('n_world_points')
            index_rms_repr_err = header.index('rms_repr_err')
            index_total_repr_err = header.index('total_repr_err')
            index_avg_rep_err = header.index('avg_rep_err_(per_point)')
            index_cam_max_t_x_err = header.index('cam_max_t_x_err')
            index_cam_max_t_y_err = header.index('cam_max_t_y_err')
            index_cam_max_t_z_err = header.index('cam_max_t_z_err')
            index_cam_mean_r_x_err = header.index('cam_mean_r_x_err')
            index_cam_mean_r_y_err = header.index('cam_mean_r_y_err')
            index_cam_mean_r_z_err = header.index('cam_mean_r_z_err')
            index_cam_max_r_x_err = header.index('cam_max_r_x_err')
            index_cam_max_r_y_err = header.index('cam_max_r_y_err')
            index_cam_max_r_z_err = header.index('cam_max_r_z_err')
            index_frame = header.index('frame')
            mean_error_t = []
            time = []
            total_time = []
            n_world_points = []
            rms_repr_err = []
            total_repr_err = []
            avg_rep_err = []
            cam_max_t_x_err = []
            cam_max_t_y_err = []
            cam_max_t_z_err = []
            mean_error_r = []
            cam_max_r_x_err = []
            cam_max_r_y_err = []
            cam_max_r_z_err = []
            frame = []
            for row in metrics_reader:
                # Shitty but I don't care at this point
                if len(row) > max(index_mean_error_t_x, index_mean_error_t_y, index_mean_error_t_z, index_time, index_total_time, index_n_world_points, index_rms_repr_err, index_total_repr_err, index_avg_rep_err, index_cam_max_t_x_err, index_cam_max_t_y_err, index_cam_max_t_z_err, index_cam_mean_r_x_err, index_cam_mean_r_y_err, index_cam_mean_r_z_err, index_cam_max_r_x_err, index_cam_max_r_y_err, index_cam_max_r_z_err, index_frame):

                    mean_error_t.append(math.sqrt(float(row[index_mean_error_t_x])**2 + float(row[index_mean_error_t_y])**2 + float(row[index_mean_error_t_z])**2))
                    time.append(float(row[index_time]))
                    total_time.append(float(row[index_total_time]))
                    n_world_points.append(int(row[index_n_world_points]))
                    rms_repr_err.append(float(row[index_rms_repr_err]))
                    total_repr_err.append(float(row[index_total_repr_err]))
                    avg_rep_err.append(float(row[index_avg_rep_err]))
                    cam_max_t_x_err.append(float(row[index_cam_max_t_x_err]))
                    cam_max_t_y_err.append(float(row[index_cam_max_t_y_err]))
                    cam_max_t_z_err.append(float(row[index_cam_max_t_z_err]))
                    mean_error_r.append(math.sqrt(float(row[index_cam_mean_r_x_err])**2 + float(row[index_cam_mean_r_y_err])**2 + float(row[index_cam_mean_r_z_err])**2))
                    cam_max_r_x_err.append(float(row[index_cam_max_r_x_err]))
                    cam_max_r_y_err.append(float(row[index_cam_max_r_y_err]))
                    cam_max_r_z_err.append(float(row[index_cam_max_r_z_err]))
                    frame.append(int(row[index_frame]))

            sample["mean_error_t"] = mean_error_t[-1]
            sample["time"] = time[-1]
            sample["total_time"] = total_time[-1]
            sample["n_world_points"] = n_world_points[-1]
            sample["rms_repr_err"] = rms_repr_err[-1]
            sample["total_repr_err"] = total_repr_err[-1]
            sample["avg_rep_err"] = avg_rep_err[-1]
            sample["cam_max_t_x_err"] = cam_max_t_x_err[-1]
            sample["cam_max_t_y_err"] = cam_max_t_y_err[-1]
            sample["cam_max_t_z_err"] = cam_max_t_z_err[-1]
            sample["mean_error_r"] = mean_error_r[-1]
            sample["cam_max_r_x_err"] = cam_max_r_x_err[-1]
            sample["cam_max_r_y_err"] = cam_max_r_y_err[-1]
            sample["cam_max_r_z_err"] = cam_max_r_z_err[-1]
            sample["frame"] = frame[-1]
            data.append(sample)

    return data

# Taken from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

# Taken from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("white", "black", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            number_as_text = valfmt(data[i, j], None)
            if number_as_text == "−−":
                number_as_text = "failed"
                color_index = 2
            else:
                color_index = int(im.norm(data[i, j]) > threshold)

            kw.update(color=textcolors[color_index])
            text = im.axes.text(j, i, number_as_text, **kw)
            texts.append(text)

    return texts

# Read the loss values from the file
list_of_files = [file for file in glob.glob('build/south_building_small/Metrics_Output*.csv')]
# Data is a list of samples. Each sample is a dictionary containing the final results and the hyperparameters
data = get_final_result_data(list_of_files)

# Plot the final mean error of t_x over all matching thresholds for nolocal BA and noglobal BA
# x = []
# y = []
# fins = True
# seen_n_times = 4
# seen_n_times_in_m_frames = 5
# localBA = False
# globalBA = False
# num_frames = 80
# for sample in data:
#     if sample["hyperparameters"]["run_local_ba"] == localBA and sample["hyperparameters"]["run_global_ba"] == globalBA and sample["hyperparameters"]["loss_function_type"] == "CAUCHY" and sample["hyperparameters"]["filter_if_not_seen"] == fins and sample["hyperparameters"]["filter_if_not_seen_n_times"] == seen_n_times and sample["hyperparameters"]["filter_if_not_seen_n_times_in_m_frames"] == seen_n_times_in_m_frames:

#         if sample["frame"] != (num_frames - 1):
#             # Reconstruction did not finish
#             print(f"Reconstruction did not finish for {sample['hyperparameters']['blurry_image_threshold']}")
#             print(f"number of frames: {sample['frame']}")
#             x.append(sample["hyperparameters"]["blurry_image_threshold"])
#             y.append(sample["mean_error_t"])
#             continue

#         x.append(sample["hyperparameters"]["blurry_image_threshold"])
#         y.append(sample["mean_error_t"])  

# # Sort the data
# x, y = zip(*sorted(zip(x, y)))
# print(f"x, y: [{x}, {y}]")

# plt.plot(x, y, label=f"localBA: {localBA}, globalBA: {globalBA}, fins: {fins}, seen_n_times: {seen_n_times}, seen_n_times_in_m_frames: {seen_n_times_in_m_frames}")
# plt.xlabel("Blurry Image Threshold")
# plt.ylabel("Mean Translational Camera Pose Error")
# plt.tight_layout()
# plt.savefig(f"figure.svg")
# plt.show()


# Print difference between local, global and no BA
# x = []
# y = []
# z = []
# z2 = []
# fins = True
# seen_n_times = 4
# seen_n_times_in_m_frames = 5
# num_frames = 128
# for sample in data:
#     if sample["hyperparameters"]["match_threshold"] == 0.7 and sample["hyperparameters"]["loss_function_type"] == "CAUCHY" and sample["hyperparameters"]["filter_if_not_seen"] == fins and sample["hyperparameters"]["filter_if_not_seen_n_times"] == seen_n_times and sample["hyperparameters"]["filter_if_not_seen_n_times_in_m_frames"] == seen_n_times_in_m_frames and sample["hyperparameters"]["blurry_image_threshold"] == 1.1:

#         if sample["frame"] != (num_frames - 1):
#             # Reconstruction did not finish
#             print(f"Reconstruction did not finish for {sample['hyperparameters']['blurry_image_threshold']}")
#             print(f"number of frames: {sample['frame']}")
#             x.append(f"localBA: {sample['hyperparameters']['run_local_ba']}, globalBA: {sample['hyperparameters']['run_global_ba']}")
#             z.append(sample["mean_error_t"])
#             z2.append(sample["mean_error_r"])
#             y.append(sample["total_repr_err"])
#             continue

#         x.append(f"localBA: {sample['hyperparameters']['run_local_ba']}, globalBA: {sample['hyperparameters']['run_global_ba']}")
#         z.append(sample["mean_error_t"])
#         y.append(sample["total_repr_err"])
#         z2.append(sample["mean_error_r"])

# # Sort the data
# # x, y = zip(*sorted(zip(x, y)))
# for i in range(len(x)):
#     print(f"BA setup: {x[i]}, avg_repr_err: {y[i]}, mean_error_t: {z[i]}, mean_error_r: {z2[i]}")



# # Print the mean translational error and rotational error for each loss function
# loss_func = []
# err_t = []
# err_r = []
# repr_err = []
# fins = True
# seen_n_times = 4
# seen_n_times_in_m_frames = 5
# num_frames = 128
# for sample in data:
#     if sample["hyperparameters"]["match_threshold"] == 0.7 and sample["hyperparameters"]["filter_if_not_seen"] == fins and sample["hyperparameters"]["filter_if_not_seen_n_times"] == seen_n_times and sample["hyperparameters"]["filter_if_not_seen_n_times_in_m_frames"] == seen_n_times_in_m_frames and sample["hyperparameters"]["blurry_image_threshold"] == 1.1:

#         print(f"frames: {sample['frame']}")

#         if sample["frame"] != (num_frames - 1):
#             # Reconstruction did not finish
#             print(f"Reconstruction did not finish for {sample['hyperparameters']['blurry_image_threshold']}")
#             print(f"number of frames: {sample['frame']}")
#             loss_func.append(f"loss: {sample['hyperparameters']['loss_function_type']}, localBA: {sample['hyperparameters']['run_local_ba']}, globalBA: {sample['hyperparameters']['run_global_ba']}")
#             err_t.append(sample["mean_error_t"])
#             err_r.append(sample["mean_error_r"])
#             repr_err.append(sample["total_repr_err"])
#             continue

#         loss_func.append(f"loss: {sample['hyperparameters']['loss_function_type']}, localBA: {sample['hyperparameters']['run_local_ba']}, globalBA: {sample['hyperparameters']['run_global_ba']}")
#         err_t.append(sample["mean_error_t"])
#         err_r.append(sample["mean_error_r"])
#         repr_err.append(sample["total_repr_err"])

# # Sort the data
# for i in range(len(loss_func)):
#     print(f"Loss function: {loss_func[i]}, mean_error_t: {err_t[i]}, mean_error_r: {err_r[i]}, total_repr_err: {repr_err[i]}")



# Heatmap of the mean translational error for match_threshold on the x-axis and filtering settings on the y-axis
# 2D numpy array
match_thresholds = [0.60, 0.65, 0.70, 0.75]
filtering_settings = [(False, 0, 0), (True, 3, 3), (True, 3, 4), (True, 4, 4), (True, 4, 5), (True, 5, 5)]
heatmap_data = np.zeros((len(match_thresholds), len(filtering_settings)))

for sample in data:
    if sample["hyperparameters"]["run_local_ba"] == False and sample["hyperparameters"]["run_global_ba"] == False and sample["hyperparameters"]["loss_function_type"] == "CAUCHY":
        match_threshold = sample["hyperparameters"]["match_threshold"]
        if match_threshold not in match_thresholds:
            continue
        filtering_setting = (sample["hyperparameters"]["filter_if_not_seen"], sample["hyperparameters"]["filter_if_not_seen_n_times"], sample["hyperparameters"]["filter_if_not_seen_n_times_in_m_frames"])
        if filtering_setting not in filtering_settings:
            continue

        match_threshold_index = match_thresholds.index(match_threshold)
        filtering_settings_index = filtering_settings.index(filtering_setting)
        heatmap_data[match_threshold_index, filtering_settings_index] = sample["mean_error_r"]

fig, ax = plt.subplots()
im, cbar = heatmap(data=heatmap_data, row_labels=match_thresholds, col_labels=["no filter", "3 in 3", "3 in 4", "4 in 4", "4 in 5", "5 in 5"], ax=ax, cbarlabel="Mean Translational Camera Pose Error", cmap="viridis")
texts = annotate_heatmap(im, valfmt="{x:.3f}")
fig.tight_layout()
fig.savefig("heatmap.png")
fig.savefig("heatmap.svg")
plt.show()