import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DNSMOS
import utils

def visualize_csv(csv_path, column=None, step=None):
    if step is not None:
        sections, ys = DNSMOS.get_label_value(csv_path, column, step)
    # only clean or not
    else:
        sections, ys = utils.get_label_value(csv_path)
    x = np.arange(len(sections))
    plt.bar(x, ys, width=1.0, align='edge')
    plt.xticks(x, sections)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, type=str)
    parser.add_argument('--column', required=False, type=str, default=None)
    parser.add_argument('--step', required=False, type=float, default=None)
    args = parser.parse_args()

    visualize_csv(args.csv_path, args.column, args.step)