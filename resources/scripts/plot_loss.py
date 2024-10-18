import glob
import re
import math
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Read the loss values from the file
list_of_files = [file for file in glob.glob('log_loss*.csv')]
print(list_of_files)

for file in list_of_files:

    title = Path(file).stem

    ba_iterations = []
    with open(title + ".csv", 'r') as file:
        losses = []
        ba_iteration = 1
        for line in file:
            row_data = line.split(',')
            cur_ba_iteration = float(row_data[0])
            loss_value = float(row_data[2])
            if cur_ba_iteration != ba_iteration:
                ba_iterations.append(losses)
                losses = []
                ba_iteration = cur_ba_iteration
            losses.append(loss_value)
        ba_iterations.append(losses)
    
    plots_per_row = 2
    num_rows = math.ceil(len(ba_iterations) / plots_per_row)
    titles = []
    for i in range(0, len(ba_iterations)):
        titles.append(f'Loss -- Iteration {i+1}')
    fig = make_subplots(rows=num_rows, cols=plots_per_row, subplot_titles=titles) 
    for i in range(0, len(ba_iterations)):
        # plt.subplot(num_rows, plots_per_row, i+1)
        # plt.xlabel(f'Bundle Adjustment Iteration {i+1}')
        # plt.ylabel('Loss')
        fig.add_trace(go.Scatter(y=ba_iterations[i]),
            row=(math.floor(i/plots_per_row)+1), col=(i%plots_per_row)+1)

    fig.update_layout(height=1600, width=800, title_text="Bundle Adjustment Loss Curves")
    # fig.show()
    fig.write_image(f"{title}.png",format='png',engine='kaleido')
    print(f"wrote file: {title}.png")