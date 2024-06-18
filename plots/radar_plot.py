# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
##### ------------------------------------------------------------------- #####

# Create the radar plot
def create_radar_plot(data, variables, title=None, colors= ['#8EC0CC', '#D6AFAE', '#83C18F']):
    """
    Creates a radar plot with the given data and variables.

    Args:
    - data: a list of tuples, where each tuple contains a label and a list of values
    - variables: a list of strings representing the variables to plot
    - title: a string representing the title of the plot (default: None)
    - colors: list of matplotlib colors

    Returns:
    - None
    """

    # Calculate angles for each variable
    num_vars = len(variables)
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    # Plot data on the radar plot
    for d, color in zip(data, colors):
        label, values = d
        values = np.array(values)
        values = np.concatenate((values, [values[0]]))
        ax.plot(theta, values, color=color, label=label)
        ax.fill(theta, values, facecolor=color, alpha=0.25)

    # Plot the reference dotted line
    ax.plot(theta, np.ones_like(theta), linestyle='--', linewidth=2, color='k', alpha=0.5)

    # Customize the plot
    ax.set_thetagrids(np.degrees(theta[:-1]), variables)
    ax.set_rgrids([1], angle=180, fontsize=10, color='k', alpha=0.5)
    # ax.spines['polar'].set_visible(False)
    ax.set_yticklabels([])

    # Add a legend and a title
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
              ncol=len(data), fontsize=14, frameon=False)

    if title is not None:
        plt.title(title, size=20, color='k', y=1.1)

    


if __name__ == '__main__':
    n = 10
    data = [
        ('Group 1', np.random.rand(n)+.5),
        ('Group 2', np.random.rand(n)+.5),
    ]
    variables = [('variable ' + str(x)) for x in range(n)]
    create_radar_plot(data, variables, colormap='plasma')
