#!/usr/bin/env python3
import glob
import os
import cooler
import numpy as np
from sklearn.preprocessing import minmax_scale
from predict_eval import get_average_preds
import pandas as pd
import argparse
from schickit.utils import get_chrom_sizes


if __name__ == '__main__':
    # Create an argument parser object
    parser = argparse.ArgumentParser(description='Aggregate single-cell predictions to get consensus loops')

    # Add mutually exclusive options, each followed by a numerical value
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--percentile", dest='percentile', type=float, help="Output the consensus loops with probability > percentile")
    group.add_argument("-n", "--num-loop", dest='num_loop', type=int, help="Output a fixed number of loops")

    # Add positional arguments
    parser.add_argument("pred_dir", type=str, help="Path to the directory storing single-cell level predictions")
    parser.add_argument("out_path", type=str, help="Path to the output file path")
    parser.add_argument("assembly_size", type=str, help="Path to the assembly sizes file (e.g. hg38.sizes)")

    # Add argument with default, providing the threshold for single cells
    parser.add_argument("-t", "--sc-threshold", type=float, default=0.5, help="Threshold for single-cell predictions")

    args = parser.parse_args()

    pred_dir = args.pred_dir
    out_path = args.out_path
    chrom_sizes_path = args.assembly_size
    sc_loop_threshold = args.sc_threshold

    base_dir = os.path.dirname(out_path)
    os.makedirs(base_dir, exist_ok=True)

    pred_files = glob.glob(os.path.join(pred_dir, '*.csv'))
    if len(pred_files) == 0:
        pred_files = glob.glob(os.path.join(pred_dir, '*.h5'))
    if len(pred_files) == 0:
        raise Exception('No prediction files found in the directory')

    if args.num_loop is not None:
        result_df = get_average_preds(
            pred_files, 10000, chrom_sizes_path,
            loop_num=args.num_loop, sc_loop_threshold=sc_loop_threshold
        )
    elif args.percentile is not None:
        result_df = get_average_preds(
            pred_files, 10000, chrom_sizes_path,
            percentile=args.percentile, sc_loop_threshold=sc_loop_threshold
        )
    else:
        raise Exception('Must specify --percentile or --num-loop')
    result_df.to_csv(out_path, sep='\t', index=False)

