import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import figure
from pathlib import Path, PurePath
from typing import Dict, Union, List
from data_juicer.utils.mm_utils import remove_special_tokens

import numpy as np
from PIL import Image


IMAGE_KEY = "images"
TEXT_KEY = "text"

def plot_dups(
    orig: dict,
    dup_list: List,
    dup_amount: int,
    scores: bool = False,
    outfile: str = None,
) -> None:
    """
    Plotting function for plot_duplicates() defined below.

    Args:
        image_dir: image directory where all files in duplicate_map are present.
        orig: filename for which duplicates are to be plotted.
        dup_list: List of duplicate filenames, could also be with scores (filename, score).
        scores: Whether only filenames are present in the dup_list or scores as well.
        outfile:  Name of the file to save the plot.
    """
    n_ims = len(dup_list)
    ncols = 4  # fixed for a consistent layout
    nrows = int(np.ceil(n_ims / ncols)) + 1
    fig = figure.Figure(figsize=(10, 14))

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    ax = plt.subplot(
        gs[0, 1:3]
    )  # Always plot the original image in the middle of top row
    
    txt = remove_special_tokens(orig[TEXT_KEY])
    orig_img = orig[IMAGE_KEY][0]
    if txt == "haha" and orig.get("type", None):
        orig_text = orig.get("type", None)
    else:
        orig_text = txt if len(txt) < 30 else txt[:30]
    dup_imgs = sum([_[IMAGE_KEY] for _ in dup_list], [])
    dup_texts = []
    for dup_each in dup_list:
        dup_txt = remove_special_tokens(dup_each[TEXT_KEY])
        if dup_txt == "haha" and dup_each.get("type", None):
            dup_texts.append(dup_each.get("type", None))
        else:
            dup_texts.append(dup_txt)
        
    ax.imshow(Image.open(orig_img))
    ax.set_title('Duplicated: %d\n%s\n%s' % (dup_amount, orig_img.split("/")[-1], orig_text), color='red', ha='center')
    ax.axis('off')

    for i in range(0, n_ims):
        row_num = (i // ncols) + 1
        col_num = i % ncols

        ax = plt.subplot(gs[row_num, col_num])
        if scores:
            ax.imshow(Image.open(dup_imgs[i][0]))
            # val = _formatter(dup_list[i][1])
            # title = ' '.join([dup_list[i][0], f'({val})'])
            # title = paths_dict[dup_list[i][0]]
        else:
            ax.imshow(Image.open(dup_imgs[i]))
            # title = dup_list[i]
            # title = paths_dict[dup_list[i]]

        ax.set_title("%s\n%s" % (dup_imgs[i].split("/")[-1], dup_texts[i]), fontsize=6)
        ax.axis('off')
    gs.tight_layout(fig)
    res = plt.gcf()
    if outfile:
        plt.savefig(outfile, dpi=300)
    plt.close()
    return res
