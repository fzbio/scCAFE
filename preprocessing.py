import pandas as pd
from schickit.bulk_hic_helper.genome_lifter import lift_bed_df
from schickit.utils import get_chrom_sizes, create_bin_df
import bioframe as bf
import numpy as np


def process_lifted_SIP_df(df, lower=100000, upper=1000000):
    df = df.reset_index(drop=True)

    df['chrom1'] = df['chrom1'].str.replace('chr', '')
    df['chrom2'] = df['chrom2'].str.replace('chr', '')

    # Flip the coordinates if x1 > y1
    flip = df['x1'] > df['y1']
    df.loc[flip, ['x1', 'x2', 'y1', 'y2']] = df.loc[flip, ['y1', 'y2', 'x1', 'x2']].values

    # Filter the coordinates
    df = df[(df['y1'] - df['x1'] >= lower) & (df['y1'] - df['x1'] <= upper)]

    # Deduplicate
    df = df.drop_duplicates(subset=['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'])
    return df


def read_SIP_bedpe(bedpe_path):
    df = pd.read_csv(
        bedpe_path, sep='\t', header=0, index_col=False,
        usecols=['chromosome1', 'x1', 'x2', 'chromosome2', 'y1', 'y2'],
    )
    df = df.rename(columns={
        'chromosome1': 'chrom1', 'chromosome2': 'chrom2'
    })
    # Flip the coordinates if x1 > y1
    flip = df['x1'] > df['y1']
    df.loc[flip, ['x1', 'x2', 'y1', 'y2']] = df.loc[flip, ['y1', 'y2', 'x1', 'x2']].values

    # Assert that all y1 > x1
    assert (df['y1'] >= df['x1']).all()
    return df


def read_fithichip_excel_as_df(excel_path, sheet_name, skiprows):
    df = pd.read_excel(
        excel_path, sheet_name=sheet_name, header=0,
        dtype={
            'Chromosome': 'str', 'Start1': 'int', 'End1': 'int',
            'Start2': 'int', 'End2': 'int'
        },
        engine='openpyxl', skiprows=skiprows
    )
    df['chrom2'] = df['Chromosome'].copy()
    df = df.rename(columns={
        'Chromosome': 'chrom1', 'Start1': 'x1', 'End1': 'x2', 'Start2': 'y1', 'End2': 'y2'
    })
    df = df[['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2']]
    return df


def read_GEO_hiccups_as_df(hiccups_path):
    df = pd.read_csv(
        hiccups_path, header=0, index_col=False, sep='\t',
        dtype={
            'chr1': 'str', 'x1': 'int', 'x2': 'int',
            'chr2': 'str', 'y1': 'int', 'y2': 'int'
        }
    )
    df['chr1'] = 'chr' + df['chr1']
    df['chr2'] = 'chr' + df['chr2']
    df = df.rename(columns={'chr1': 'chrom1', 'chr2': 'chrom2'})
    df = df[['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2']]
    return df


def read_full_insulation_df_from_bed(bed_path):
    # Difference with read_tad_score_from_bed: this function returns all columns of the bed file, and
    # does not change it to 0-based.
    tad_score_df = pd.read_csv(
        bed_path, sep='\t', header=None, names=['chrom', 'start', 'end', 'dot1', 'score', 'dot2']
    )
    return tad_score_df



def calculate_overlap_length_row(row):
    start1 = row['start']
    end1 = row['end']
    start2 = row['start_']
    end2 = row['end_']
    overlap_length = min(end1, end2) - max(start1, start2)
    if overlap_length < 0:
        return 0
    else:
        return overlap_length


def assign_lifted_insulation_to_binned_df(binned_df, lifted_insulation_df):
    overlap_df = bf.overlap(
        lifted_insulation_df, binned_df, how='left',
        cols1=('chrom', 'start', 'end'), cols2=('chrom', 'start', 'end')
    )
    overlap_df['overlap_length'] = overlap_df.apply(calculate_overlap_length_row, axis=1)
    df = overlap_df.drop(['chrom', 'start', 'end'], axis=1)
    df = df.rename(columns={'chrom_': 'chrom', 'start_': 'start', 'end_': 'end'})
    return df




def lift_fanc_insulation_score(insulation_path, target_chrom_size_path, liftover_exec_path, chain_path, tmp_dir_path, resolution=10000):
    """
    Lift the insulation score from one assembly
    """
    # Read the insulation score
    tad_score_df = read_full_insulation_df_from_bed(insulation_path)

    # Convert the insulation score to 0-based
    tad_score_df['start'] = tad_score_df['start'] - 1

    # Lift the insulation score
    tad_score_df = lift_bed_df(tad_score_df, liftover_exec_path, chain_path, tmp_dir_path)

    # Assign the lifted insulation score to the binned df (not genome-wide, but only the lifted regions)
    bin_df = create_bin_df(target_chrom_size_path, resolution)
    tad_score_df = assign_lifted_insulation_to_binned_df(bin_df, tad_score_df)

    # Aggregate the insulation score by weighted average
    def nan_average(x):
        a = x['score']
        weights = x['overlap_length']
        ma = np.ma.masked_array(a, np.isnan(a))
        return pd.Series(np.ma.average(ma, weights=weights), index=['score'])
    tad_score_df = tad_score_df.groupby(['chrom', 'start', 'end', 'dot1', 'dot2'], as_index=False).apply(
        nan_average
    ).reset_index(drop=True)

    # create the (genome-wide) new bin df to store the lifted insulation score
    existing_chrom_names = sorted(list(tad_score_df['chrom'].unique()))
    genome_wide_bin_df = create_bin_df(target_chrom_size_path, resolution, chrom_order=existing_chrom_names)
    tad_score_df = genome_wide_bin_df.merge(tad_score_df, how='left', on=['chrom', 'start', 'end'], validate='one_to_one')
    tad_score_df = tad_score_df.reset_index(drop=True)

    # Convert back the insulation score to 1-based
    tad_score_df['start'] = tad_score_df['start'] + 1

    # Fill nan dots with '.'
    tad_score_df['dot1'] = tad_score_df['dot1'].fillna('.')
    tad_score_df['dot2'] = tad_score_df['dot2'].fillna('.')
    assert len(tad_score_df['dot1'].unique()) == 1

    return tad_score_df[['chrom', 'start', 'end', 'dot1', 'score', 'dot2']]


if __name__ == '__main__':
    df = lift_fanc_insulation_score('test_data/test_hg38.bed', 'external_annotations/hg19.sizes', 'hg38', 'hg19')
    print(df)
    print(df[~df['score'].isna()])