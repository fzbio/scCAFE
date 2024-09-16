import cooler
import numpy as np
import pandas as pd


def check_cool_duplicate_pixels(cool_path):
    clr = cooler.Cooler(cool_path)
    pixels = clr.pixels()[:]
    # Check if there are any duplicates in pixels dataframe
    result = pixels.duplicated(subset=['bin1_id', 'bin2_id'], keep=False).any()
    return result


if __name__ == '__main__':
    print(check_cool_duplicate_pixels('/home/fuzhou/hic_research/sc-hic-loop/plot_scripts/plot_data/mES_bulk_mm10.mcool::/resolutions/100000'))
