import glob
import re
import math
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Read the loss values from the file
list_of_files = [file for file in glob.glob('reprojection_error*.csv')]
print(list_of_files)

for file in list_of_files:

    title = Path(file).stem

    with open(title + ".csv", 'r') as file:

        fig = go.Figure()
        mean_reproject_errors = { 'no_ba': [], 'ba': [] }
        for line in file:
            row_data = line.split(',')
            cur_frame = float(row_data[1])
            mean_reproject_error = float(row_data[2])
            if row_data[0] == 'no_ba':
                mean_reproject_errors['no_ba'].append(mean_reproject_error)
            if row_data[0] == 'ba':
                mean_reproject_errors['ba'].append(mean_reproject_error)
        
        # add correct labels in case of 'ba' vs. 'no ba'
        if 'ba' in mean_reproject_errors:
            fig.add_trace(go.Scatter(y=mean_reproject_errors['ba'], name='mean reproject error (after BA)'))
            fig.add_trace(go.Scatter(y=mean_reproject_errors['no_ba'], name='mean reproject error (before BA)'))
        else:
            fig.add_trace(go.Scatter(y=mean_reproject_errors['no_ba'], name='mean reproject error')) 

        fig.update_layout(height=800, width=800, title_text="Reprojection Error Curve", xaxis_title="frame counter", yaxis_title="mean reproject error")
        # fig.show()
        fig.write_image(f"{title}.png",format='png',engine='kaleido')
        print(f"wrote file: {title}.png")