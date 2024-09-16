import cooler
import os
from .file_format_conversion import convert_cool_to_scool
from tqdm.auto import tqdm
import time
from tempfile import TemporaryDirectory
import pandas as pd
import pyfastx
import numpy as np
import cooler
from bioframe.extras import binnify


CHUNK_SIZE = 1000000


def get_cg_content_bin_df(fasta_path, chrom_names, resolution, chrom_sizes):
    fasta = pyfastx.Fasta(fasta_path)
    bins_df = create_bin_df(chrom_sizes, resolution, chrom_names)
    def get_locus_gc(row):
        chrom_name = row['chrom']
        seq = fasta[chrom_name][row['start']:row['end']]
        gc = seq.gc_content
        if np.isnan(gc):
            gc = 0
        return gc
    bins_df = bins_df[bins_df['chrom'].isin(chrom_names)]
    bins_df = bins_df.reset_index(drop=True)
    bins_df['gc'] = bins_df.apply(get_locus_gc, axis=1)
    return bins_df



def create_bin_df(chrom_sizes, resolution, chrom_order=None):
    if isinstance(chrom_sizes, str):
        chrom_sizes = get_chrom_sizes(chrom_sizes)
    if chrom_order is None:
        chrom_order = sorted(chrom_sizes.keys())
    chrom_size_series = pd.Series(chrom_sizes, index=chrom_order)
    return binnify(chrom_size_series, resolution)


def get_chrom_sizes(file_path):
    sizes = {}
    with open(file_path, 'r') as fp:
        for line in fp:
            line_split = line.split()
            sizes[line_split[0]] = int(line_split[1])
    return sizes


def coarsen_scool(scool_path, out_scool_path, parse_func):
    # This function is problematic, because we do not specify the coarsen factor in it, and the suffix of the
    # intermediate cool file is always 100kb, which will potentially cause problems in real-world applications.
    print('Coarsening data...')
    with TemporaryDirectory() as temp_dir:
        cell_list = cooler.fileops.list_scool_cells(scool_path)
        for cell_path in tqdm(cell_list):
            cell_uri = scool_path + '::' + cell_path
            # print(cell_uri)
            cell_name = cell_uri.split('/')[-1]
            cooler.coarsen_cooler(cell_uri, os.path.join(temp_dir, cell_name) + '_100kb_contacts.cool', 10, 1000000, 12)
        convert_cool_to_scool(temp_dir, out_scool_path, parse_func)
    print('Done!')


def get_bin_count(base_count, resolution):
    if base_count % resolution != 0:
        return int(base_count / resolution) + 1
    else:
        return int(base_count / resolution)


def grouplist(L, grp_size):
    starts = range(0, len(L), grp_size)
    stops = [x + grp_size for x in starts]
    groups = [L[start:stop] for start, stop in zip(starts, stops)]
    return groups


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return timed
