import pandas as pd
import os
import shutil
import glob


def filter_range_for_df(df, lower=100000, higher=1000000):
    return df[
        (df['y1'] - df['x1'] >= lower) & (df['y1'] - df['x1'] <= higher)
    ]


def align_promoter_coords_to_locus(promoter_df, resolution):
    promoter_df.loc[:, 'locus'] = ((promoter_df['end'] + promoter_df['start']) // 2) // resolution * resolution
    return promoter_df

def find_promoter_coords_from_gene_coords(gene_coords_df):
    promoter_coords = gene_coords_df.copy()
    promoter_starts = []
    for start, end, strand in zip(gene_coords_df['start'], gene_coords_df['end'], gene_coords_df['strand']):
        if strand == '+':
            promoter_starts.append(start - 2000)
        else:
            promoter_starts.append(end)
    promoter_coords['start'] = promoter_starts
    promoter_coords['end'] = promoter_coords['start'] + 2000
    return promoter_coords


def remove_existing_scool(scool_path):
    if os.path.exists(scool_path):
        os.remove(scool_path)


def up_lower_tria_vote(df):
    df = df[df['chrom1'] == df['chrom2']]
    triu_df = df[df['y1'] > df['x1']]
    tril_df = df[df['y1'] < df['x1']]
    tril_df = tril_df.rename(columns={'x1': 'y1', 'x2': 'y2', 'y1': 'x1', 'y2': 'x2'})
    tril_df = tril_df[['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2', 'proba']]
    df = pd.concat([triu_df, tril_df])
    df = df.groupby(['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], as_index=False, sort=False).mean()
    return df


def deduplicate_bedpe_dir(bedpe_dir):
    bedpe_files_paths = glob.glob(os.path.join(bedpe_dir, '*.csv'))
    for bedpe_path in bedpe_files_paths:
        df = pd.read_csv(bedpe_path, sep='\t', header=0, index_col=False, dtype={'proba': 'float'})
        try:
            df = df.groupby(['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], sort=True, as_index=False).mean()
        except:
            print(df)
        df.to_csv(bedpe_path, sep='\t', header=True, index=False)


def remove_datasets(dir_list):
    for d in dir_list:
        shutil.rmtree(d)


def get_loop_calling_dataset_paths(graph_ds_dir, loop_calling_ds_name):
    train_name = loop_calling_ds_name + '_train'
    val_name = loop_calling_ds_name + '_val'
    test_name = loop_calling_ds_name + '_test'
    train_path = os.path.join(graph_ds_dir, train_name)
    val_path = os.path.join(graph_ds_dir, val_name)
    test_path = os.path.join(graph_ds_dir, test_name)
    return train_path, val_path, test_path


def get_imputes_dataset_paths(graph_ds_dir, imputation_ds_name):
    return get_loop_calling_dataset_paths(graph_ds_dir, imputation_ds_name)


def get_split_scool_paths(store_dir, store_scool_name):
    train_name = store_scool_name + '.train.scool'
    val_name = store_scool_name + '.val.scool'
    test_name = store_scool_name + '.test.scool'
    train_path = os.path.join(store_dir, train_name)
    val_path = os.path.join(store_dir, val_name)
    test_path = os.path.join(store_dir, test_name)
    return train_path, val_path, test_path


def neuron_celltype_parser(cell_name, desired_ctypes):
    cell_type = cell_name.split('_')[-1]
    if cell_type in ['L23', 'L4', 'L5', 'L6', 'Sst', 'Vip', 'Ndnf', 'Pvalb']:
        ct = 'Neuron'
    else:
        return None

    if ct in desired_ctypes:
        return ct
    else:
        return None


def hpc_celltype_parser(cell_name, desired_ctypes):
    # print(cell_name)
    cell_type = cell_name.split('_')[-1]
    if cell_type == 'MG':
        ct = 'MG'
    elif cell_type == 'ODC':
        ct = 'ODC'
    elif cell_type in ['L23', 'L4', 'L5', 'L6', 'Sst', 'Vip', 'Ndnf', 'Pvalb']:
        ct = 'Neuron'
    else:
        return None

    if ct in desired_ctypes:
        return ct
    else:
        return None


def read_umicount_as_df(unicount_path):
    # Return a dataframe where each ROW is the gene expression of A CELL.
    # NOTE: This is the transpose of the original umicount file.
    df = pd.read_csv(unicount_path, sep='\t', index_col='gene')
    df = df.transpose()
    df.index.names = ['cell']
    df.columns.names = ['gene']
    return df


def read_meta_data(meta_data_path):
    df = pd.read_csv(meta_data_path, sep='\t', index_col=False, header=0)
    return df


def get_cell_name_attribute_dict_from_metadata(meta_df, attribute):
    the_dict = dict(zip(meta_df['Cellname'], meta_df[attribute]))
    return the_dict