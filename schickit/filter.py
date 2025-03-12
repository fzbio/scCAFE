import cooler
import numpy as np
from fontTools.subset import subset
from tqdm.auto import tqdm
import pandas as pd
import tempfile
import os
from .file_format_conversion import convert_cool_list_to_scool
from tqdm.auto import tqdm


def repeat_rows_by_count(df):
    repeated_indices = np.repeat(df.index, df["count"])
    repeated_df = df.loc[repeated_indices].reset_index(drop=True)
    return repeated_df


def downsample_cooler_to_create_scool(cool_path, out_scool_path, ref_scool_path, num_cells, resolution, tmp_dir):
    ref_scool_cell_names = cooler.fileops.list_scool_cells(ref_scool_path)
    assert num_cells <= len(ref_scool_cell_names)
    ref_scool_cell_names = np.random.choice(ref_scool_cell_names, num_cells, replace=False)
    clr = cooler.Cooler(cool_path)
    chrom_num = len(clr.chromnames)
    bins = clr.bins()[:]
    bin_count = bins.shape[0]
    pixels = clr.pixels()[:]
    pixels_join = clr.pixels(join=True)[:]
    pixels_join['bin1_id'] = pixels['bin1_id']
    pixels_join['bin2_id'] = pixels['bin2_id']
    pixels = pixels_join
    pixels = pixels[(pixels['chrom1'] == pixels['chrom2'])]
    max_dist = np.max(pixels['start2'] - pixels['start1']) // resolution
    max_dist = np.min([max_dist, 2000])
    mean_count_each_dist = []
    pixels_each_dist = []
    print('Counting contacts for each distance...')
    for dist in tqdm(range(max_dist + 1)):
        current_dist_pixels = pixels[(pixels['start2'] - pixels['start1']) // resolution == dist]
        pixels_each_dist.append(current_dist_pixels)
        count = np.sum(current_dist_pixels['count']) if len(current_dist_pixels) > 0 else 0
        mean_count_each_dist.append(count / (bin_count - dist * chrom_num))
    mean_count_each_dist = np.array(mean_count_each_dist)
    sanitizer = cooler.create.sanitize_pixels(clr.bins()[:], sort=True, tril_action='raise')
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tempdir:
        temp_cool_path_list = []
        print('Creating cooler files...')
        ref_bin_count = None
        ref_chrom_num = None
        for i, ref_scool_cell_name in enumerate(tqdm(ref_scool_cell_names)):
            temp_cooler_path = os.path.join(tempdir, f'tmp_{i}.cool')
            ref_clr = cooler.Cooler(ref_scool_path + '::' + ref_scool_cell_name)
            if ref_bin_count is None:
                ref_bin_count = ref_clr.bins()[:].shape[0]
                ref_chrom_num = len(ref_clr.chromnames)
            ref_pixels_join = ref_clr.pixels(join=True)[:]
            ref_pixels_join = ref_pixels_join[(ref_pixels_join['chrom1'] == ref_pixels_join['chrom2'])]
            ref_max_dist = np.max(ref_pixels_join['start2'] - ref_pixels_join['start1']) // resolution
            current_cell_max_dist = np.min([max_dist, ref_max_dist])
            current_cell_pixels = []
            for dist in range(current_cell_max_dist + 1):
                mean_count = mean_count_each_dist[dist]
                current_dist_ref_pixels = ref_pixels_join[(ref_pixels_join['start2'] - ref_pixels_join['start1']) // resolution == dist]
                ref_mean_count = np.sum(current_dist_ref_pixels['count']) if len(current_dist_ref_pixels) > 0 else 0
                ref_mean_count = ref_mean_count / (ref_bin_count - dist * ref_chrom_num)
                sampling_proba = ref_mean_count / mean_count
                assert sampling_proba <= 1
                current_dist_pixels = pixels_each_dist[dist]
                current_dist_pixels = repeat_rows_by_count(current_dist_pixels)
                current_dist_pixels['count'] = 1
                current_dist_pixels = current_dist_pixels.sample(frac=sampling_proba)
                current_dist_pixels = current_dist_pixels[['bin1_id', 'bin2_id', 'count']]

                # Remove some pixels and add noise to current_dist_pixels
                remove_mask = np.random.rand(len(current_dist_pixels)) < 0.1
                current_dist_pixels = current_dist_pixels[remove_mask]
                noise_bins_indices = np.random.randint(0, bin_count - dist, size=len(current_dist_pixels) * 9)
                noise_bins = bins.iloc[noise_bins_indices]
                noise_pixels = pd.DataFrame({
                    'bin1_id': noise_bins.index,
                    'bin2_id': noise_bins.index + dist,
                    'count': 1
                })
                current_dist_pixels = pd.concat([current_dist_pixels, noise_pixels])

                current_dist_pixels = current_dist_pixels.groupby(['bin1_id', 'bin2_id'], as_index=False).sum()
                current_cell_pixels.append(current_dist_pixels)
            current_cell_pixels = pd.concat(current_cell_pixels).reset_index(drop=True)
            current_cell_pixels = sanitizer(current_cell_pixels)
            cooler.create_cooler(temp_cooler_path, clr.bins()[:], current_cell_pixels)
            temp_cool_path_list.append(temp_cooler_path)
        convert_cool_list_to_scool(temp_cool_path_list, out_scool_path, lambda x: x.split('/')[-1][:-5])






# def downsample_cooler_to_create_scool(cool_path, out_scool_path, num_pixels_per_chrom, num_cells, tmp_dir):
#     clr = cooler.Cooler(cool_path)
#     pixels = clr.pixels()[:]
#     pixels['count'] = np.round(np.log2(pixels['count']) + 1).astype('int')
#     pixels = repeat_rows_by_count(pixels)
#     pixels['count'] = 1
#     chroms = clr.chromnames
#     sc_num_pixels = num_pixels_per_chrom * len(chroms)
#     divide_cool_into_n_total = len(pixels) // sc_num_pixels
#     assert divide_cool_into_n_total >= num_cells
#     print(divide_cool_into_n_total)
#     random_labels = np.random.randint(0, divide_cool_into_n_total, size=len(pixels))
#     sanitizer = cooler.create.sanitize_pixels(clr.bins()[:], sort=True, tril_action='raise')
#     with tempfile.TemporaryDirectory(dir=tmp_dir) as tempdir:
#         temp_cool_path_list = []
#         for i in tqdm(range(num_cells)):
#             temp_cooler_path = os.path.join(tempdir, f'tmp{i}.cool')
#             current_cell_pixels = pixels[random_labels == i].reset_index(drop=True)
#             current_cell_pixels = current_cell_pixels.groupby(['bin1_id', 'bin2_id'], as_index=False).sum()
#             current_cell_pixels = current_cell_pixels.reset_index(drop=True)
#             current_cell_pixels = sanitizer(current_cell_pixels)
#             cooler.create_cooler(temp_cooler_path, clr.bins()[:], current_cell_pixels)
#             temp_cool_path_list.append(temp_cooler_path)
#         convert_cool_list_to_scool(temp_cool_path_list, out_scool_path, lambda x: x.split('/')[-1][:-5])


def count_contacts_in_scool(scool_path):
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    name_count_dict = {}
    for cell_name in cell_names:
        clr = cooler.Cooler(scool_path + "::" + cell_name)
        count = clr.info['sum']
        name_count_dict[cell_name] = count
    return name_count_dict


def filter_cells(scool_path, left_cell_num, out_path):
    name_count_dict = count_contacts_in_scool(scool_path)
    counts = np.array(list(name_count_dict.values()))
    threshold = np.partition(counts, -left_cell_num)[-left_cell_num]
    for name in tqdm(name_count_dict):
        if name_count_dict[name] >= threshold:
            cool_path = scool_path + '::' + name
            clr = cooler.Cooler(cool_path)
            bins = clr.bins()[:]
            pixels = clr.pixels()[:]
            cooler.create_scool(out_path, {name: bins}, {name: pixels}, mode='a', symmetric_upper=True)
