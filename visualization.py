import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import DNSMOS
from . import utils

def visualize_csv(csv_path, column=None, step=None):
    if step is not None:
        sections, ys = DNSMOS.get_label_value(csv_path, column, step)
        width = 1.0
        align = 'edge'
    # only clean or not
    else:
        sections, ys = utils.get_label_value(csv_path)
        width = 0.5
        align = 'center'
    
    x = np.arange(len(sections))

    plt.bar(x, ys, width=width, align=align)
    plt.xticks(x, sections)
    plt.show()

def visualize_drift(csv_dir, state=None, order=0):
    state_counts = utils.get_state_ratio(csv_dir)
    ys = utils.apply_differential(state_counts[state], order)
    x = np.arange(len(ys))

    plt.plot(x, ys)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, type=str)
    parser.add_argument('--column', required=False, type=str, default=None)
    parser.add_argument('--step', required=False, type=float, default=None)
    args = parser.parse_args()

    visualize_csv(args.csv_path, args.column, args.step)