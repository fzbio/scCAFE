import cooler
import os
import re
from tqdm.auto import tqdm
import pandas as pd
import random


# def convert_cool_to_scool(cool_dir, scool_path, parse_func):
#     cool_paths = [os.path.join(cool_dir, e) for e in os.listdir(cool_dir)]
#     # print(cool_paths)
#     for cool_path in tqdm(cool_paths):
#         cell_name = parse_func(cool_path)
#         clr = cooler.Cooler(cool_path)
#         bins = clr.bins()[:]
#         pixels = clr.pixels()[:]
#         cooler.create_scool(scool_path, {cell_name: bins}, {cell_name: pixels}, mode='a', symmetric_upper=True)
#         # print(cell_name)


def convert_cool_list_to_scool(cool_paths, scool_path, parse_func):
    cell_name = parse_func(cool_paths[0])
    clr = cooler.Cooler(cool_paths[0])
    bins = clr.bins()[:]
    pixels = clr.pixels()[:]
    cooler.create_scool(scool_path, {cell_name: bins}, {cell_name: pixels}, mode='w', symmetric_upper=True, dupcheck=True, triucheck=True)
    for cool_path in cool_paths[1:]:
        cell_name = parse_func(cool_path)
        cooler.fileops.cp(cool_path, scool_path + '::/cells/' + cell_name)


def convert_cool_to_scool(cool_dir, scool_path, parse_func):
    cool_paths = [os.path.join(cool_dir, e) for e in os.listdir(cool_dir)]
    cell_name = parse_func(cool_paths[0])
    clr = cooler.Cooler(cool_paths[0])
    bins = clr.bins()[:]
    pixels = clr.pixels()[:]
    cooler.create_scool(scool_path, {cell_name: bins}, {cell_name: pixels}, mode='w', symmetric_upper=True, dupcheck=True, triucheck=True)
    for cool_path in cool_paths[1:]:
        cell_name = parse_func(cool_path)
        cooler.fileops.cp(cool_path, scool_path + '::/cells/' + cell_name)


def aggregate_scool(scool_path, output_cool_path, name_determine_func=None):
    if name_determine_func is None:
        cell_names = cooler.fileops.list_scool_cells(scool_path)
    else:
        all_cell_names = cooler.fileops.list_scool_cells(scool_path)
        cell_names = []
        for cn in all_cell_names:
            if name_determine_func(cn):
                cell_names.append(cn)
    cool_uris = [scool_path + '::' + c for c in cell_names]
    cooler.merge_coolers(output_cool_path, cool_uris, 100000000)


if __name__ == '__main__':
    # df0,  df1 = convert_mES_excel_to_bedpe('../data/mES/mES_bulk_loop.xlsx', None, ['Bulk_HiC_filter', 'H3K4me3_PLACseq_filter'])
    # print(len(df0), len(df1))
    pass

