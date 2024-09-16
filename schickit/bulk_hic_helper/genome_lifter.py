from pyliftover import LiftOver
import pandas as pd
import tempfile
import os
import numpy as np


def convert_genomic_coord(liftover_obj, chrom, coord):
    # Only convert one coordinate. Not valid for genomic intervals.
    lifted_coord_list = liftover_obj.convert_coordinate(chrom, coord)
    if lifted_coord_list is None:
        raise Exception
    elif len(lifted_coord_list) >= 1:
        lifted_coord = lifted_coord_list[0][1]
        lifted_chrom = lifted_coord_list[0][0]
        if lifted_chrom != chrom:
            return ['chr999', -1]
        else:
            return [chrom, lifted_coord]
    else:
        return ['chr999', -1]


def lift_bed_df(df, liftover_exec_path, chain_path, tmp_root):
    # This function can accept any bed-like df, but the df must have 'chrom', 'start', 'end' columns.
    # Convert the df with liftover BED4 format.
    # No -multiple allowed.
    with tempfile.TemporaryDirectory(dir=tmp_root) as temp_dir:
        entry_id = np.arange(len(df))
        df_bed4 = df[['chrom', 'start', 'end']].copy()
        df_other = df.drop(['chrom', 'start', 'end'], axis=1).copy()
        df_bed4['entry_id'] = entry_id.copy()
        df_other['entry_id'] = entry_id.copy()

        input_bed_path = os.path.join(temp_dir, 'input.bed')
        output_bed_path = os.path.join(temp_dir, 'output.bed')
        failed_bed_path = os.path.join(temp_dir, 'unlifted.bed')


        df_bed4.to_csv(input_bed_path, sep='\t', header=False, index=False, na_rep='nan')
        os.system(f'{liftover_exec_path} {input_bed_path} {chain_path} {output_bed_path} {failed_bed_path}')
        lifted_df_bed4 = pd.read_csv(
            output_bed_path, sep='\t', header=None,
            names=df_bed4.columns.tolist()
        )
        lifted_df = lifted_df_bed4.merge(df_other, on='entry_id', validate='one_to_one')
        lifted_df = lifted_df.drop('entry_id', axis=1)
        lifted_df = lifted_df[df.columns.tolist()]
        lifted_df = lifted_df.reset_index(drop=True)
        return lifted_df


def lift_paired_df(df, liftover_exec_path, chain_path, tmp_root):
    with tempfile.TemporaryDirectory(dir=tmp_root) as temp_dir:
        contact_id = np.arange(len(df))
        df_left = df[['chrom1', 'x1', 'x2']].copy()
        df_right = df[['chrom2', 'y1', 'y2']].copy()
        df_other = df.drop(['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], axis=1).copy()
        df_left['contact_id'] = contact_id.copy()
        df_right['contact_id'] = contact_id.copy()
        df_other['contact_id'] = contact_id.copy()
        df_left.to_csv(os.path.join(temp_dir, 'left.bed'), sep='\t', header=False, index=False, na_rep='nan')
        df_right.to_csv(os.path.join(temp_dir, 'right.bed'), sep='\t', header=False, index=False, na_rep='nan')
        os.system(
            f'{liftover_exec_path} {temp_dir}/left.bed {chain_path} {temp_dir}/left_lifted.bed {temp_dir}/left_unlifted.bed'
        )
        os.system(
            f'{liftover_exec_path} {temp_dir}/right.bed {chain_path} {temp_dir}/right_lifted.bed {temp_dir}/right_unlifted.bed'
        )
        left_lifted_df = pd.read_csv(
            os.path.join(temp_dir, 'left_lifted.bed'), sep='\t', header=None,
            names=['chrom1', 'x1', 'x2', 'contact_id']
        )
        right_lifted_df = pd.read_csv(
            os.path.join(temp_dir, 'right_lifted.bed'), sep='\t', header=None,
            names=['chrom2', 'y1', 'y2', 'contact_id']
        )
        # As liftover -multiple is not used, there should not be any duplicated contact_id.
        lifted_df = pd.merge(left_lifted_df, right_lifted_df, on='contact_id', validate='one_to_one')
        lifted_df = pd.merge(lifted_df, df_other, on='contact_id', validate='one_to_one')
        lifted_df = lifted_df.drop('contact_id', axis=1)
        lifted_df = lifted_df[df.columns.tolist()]
        lifted_df = lifted_df.reset_index(drop=True)
        return lifted_df
