import json
import os
from itertools import groupby

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from custom_methods import get_parser


def plot_percentage(df, x_var, ax):
    """
    100% taken from https://stackoverflow.com/a/67076347, many thanks
    args:
        df: pandas dataframe
        x_var: (string) X variable
        ax: Axes object (for Seaborn Countplot/Bar plot or
                         pandas bar plot)
    """
    # 1. how many X categories
    # check for NaN and remove
    num_x = len([x for x in df[x_var].unique() if x == x])

    # 2. The bars are created in hue order, organize them
    bars = ax.patches
    # 2a. For each X variable
    for ind in range(num_x):
        # 2b. Get every hue bar
        # ex. 8 X categories, 4 hues =>
        # [0, 8, 16, 24] are hue bars for 1st X category
        hue_bars = bars[ind:][::num_x]
        # 2c. Get the total height (for percentages)
        total = sum([x.get_height() for x in hue_bars])
        if math.isnan(total):
            continue
        # 3. Print the percentage on the bars
        for bar in hue_bars:
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height(),
                    f'{bar.get_height() / total:.1%}',
                    ha="center", va="bottom", fontsize='xx-small')


def plot_title(json_name):
    """
    Jank function to make the graphs prettier and consistent with report
    :param json_name: name of the json that is being plotted
    :return: graph title
    """
    translate = {"onlyme": "Excavators inhouse", "exc": "Excavators", "fork": "Forklift", "scissor": "Scissor lifts",
                 "combined": "Combined"}
    for key in translate:
        if key in json_name:
            return translate[key] + " " + json_name.split('_')[-1]
    return json_name


def main(input_path):
    json_name = input_path.split('/')[-1]
    json_name = json_name.split('.')[0]

    # Load the json
    print('Loading json file...')
    with open(input_path) as train_json_file:
        train_json = json.load(train_json_file)

    # count annotations per category
    category_id_names = {category["id"]: category["supercategory"] for category in train_json["categories"]}
    # print("ID to name for all the categories:")
    # print(category_id_names)

    train_counts = dict.fromkeys(category_id_names.values(), 0)
    train_sizes = []
    for category, annotations in groupby(train_json["annotations"], lambda x: x["category_id"]):
        # convert category id to category name
        category_name = category_id_names.get(category)
        for annotation in annotations:
            # count entries
            train_counts[category_name] += 1
            # determine number for each size
            # 32x32 APs 96x96 APm >96x96 APl
            if annotation["area"] <= 32 * 32:
                size = "small"
            elif 96 * 96 >= annotation["area"] > 32 * 32:
                size = "medium"
            else:
                size = "large"
            train_sizes.append({"category": category_name, "size": size})
            train_sizes.append({"category": "total", "size": size})

    train_sizes = sorted(train_sizes, key=lambda k: k["category"])
    train_sizes = pd.DataFrame(train_sizes)

    # print some statistics about the data
    print("Annotations per category: ")
    print(train_counts)
    print(f"With a total of {sum(train_counts.values())}.")
    print("More detailed statistics: ")
    print(train_sizes.value_counts())

    # filter out the very low frequency categories
    m = 0.02 * len(train_sizes)
    for c in train_sizes.columns:
        train_sizes = train_sizes[
            train_sizes[c].isin(train_sizes[c].value_counts()[train_sizes[c].value_counts() > m].index)]

    # print some more statistics after filtering
    print(f"After filtering out lower than 1% of total this leaves: ")
    print(train_sizes.value_counts())

    # set order to make it equal for each json
    class_order = train_sizes["category"].unique()
    hue_order = sorted(train_sizes["size"].unique())

    # set style for spines
    sns.set_style("ticks", {'axes.linewidth': 2, 'axes.edgecolor': 'black'})
    g = sns.catplot(x="category", hue="size", data=train_sizes, kind='count', facet_kws=dict(despine=False),
                    order=class_order, hue_order=hue_order)

    # set title
    g.fig.suptitle(plot_title(json_name) + " distribution")
    # make ticks smaller
    g.set_xticklabels(fontsize=8)
    # plot percentage on top of bars
    plot_percentage(train_sizes, 'category', g.ax)
    # fix the whitespaces
    plt.subplots_adjust(left=0.12, bottom=0.11, top=0.88)

    # check output path and save fig
    output_dir = f"{args.dir}graphs/"
    plot_name = f"{output_dir}{json_name}.png"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    plt.savefig(fname=plot_name, dpi=600)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    # if single file do that, else take entire folder
    if args.filename:
        main(args.filename)
    else:
        for filename in os.listdir(args.dir):
            if filename.endswith(".json"):
                main(args.dir + filename)
