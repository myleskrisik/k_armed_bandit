import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('build/figures.json') as f:
    figures = json.load(f)
    for figure in figures['figures']:
        x_label = figure['x_label']
        y_label = figure['y_label']

        data_frames = []
        for plot in figure['plots']:
            plot_label = plot['label']
            x_data = plot['x_data']
            y_data = plot['y_data']
            data_frame = pd.DataFrame({x_label: x_data, y_label: y_data, 'method': plot_label})
            data_frames.append(data_frame)

        line_plot = sns.lineplot(x= x_label, y = y_label, data = pd.concat(data_frames), hue = 'method', errorbar = None)
        fig = line_plot.get_figure()
        fig.savefig(f"images/{figure['title']}.png")
        plt.close(fig)
    
