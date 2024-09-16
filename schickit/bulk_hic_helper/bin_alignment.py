import pandas as pd
from ..utils import get_chrom_sizes


def align_paired_df_to_resolution(df, resolution):
    # This function also needs to deal with the largest coordinate in each chromosome in the future.
    df = df.copy()
    x_middle = (df['x1'] + df['x2']) // 2
    y_middle = (df['y1'] + df['y2']) // 2
    df['x1'] = x_middle // resolution * resolution
    df['x2'] = df['x1'] + resolution
    df['y1'] = y_middle // resolution * resolution
    df['y2'] = df['y1'] + resolution
    return df


def align_bed_df_to_resolution(df, resolution, chrom_sizes_path):
    df = df.copy()
    middle = (df['start'] + df['end']) // 2
    df['start'] = middle // resolution * resolution
    df['end'] = df['start'] + resolution
    # Check if there are coordinates larger than chrom sizes for each chromosome
    chrom_sizes = get_chrom_sizes(chrom_sizes_path)
    for chrom in chrom_sizes:
        chrom_size = chrom_sizes[chrom]
        if len(df[df['chrom'] == chrom]) > 0:
            assert df[df['chrom'] == chrom]['start'].max() <= chrom_size
            assert len(df[df['chrom'] == chrom][df['end'] > chrom_size]) in [0, 1]
            if len(df[df['chrom'] == chrom][df['end'] > chrom_size]) == 1:
                # Find the index of the highest coordinate
                df.loc[(df['chrom'] == chrom) & (df['end'] > chrom_size), 'end'] = chrom_size
                print(df[(df['chrom'] == chrom) & (df['end'] > chrom_size)])
    return df
