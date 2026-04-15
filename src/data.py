#-----------------------------------------------------------
#                         data.py
#-----------------------------------------------------------

import os
from os.path import join, basename
import os.path as osp
import glob
import torch
import gc
from glob import iglob
from torch_geometric.data import Dataset
from torch.utils.data import random_split
from os.path import join
from result import Result
import json
import math
from math import ceil
from shutil import rmtree
from scipy.sparse import hstack, coo_matrix, csr_matrix
from utils import get_root_path, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, load, save
from config import FLAGS, ALL_KERNEL
from saver import saver
import networkx as nx
from tqdm import tqdm

from math import ceil

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, Batch

import networkx as nx
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack, coo_matrix, csr_matrix

import os.path as osp
import torch
from torch_geometric.data import Dataset
from torch.utils.data import random_split

from shutil import rmtree
import math
from collections import OrderedDict, defaultdict, Counter
import csv


APP_INFO_CSV = join(get_root_path(),
                    'Data4LLMPrompting',
                    'ApplicationInformation.csv')
APL_MAPPING_DIR = join(get_root_path(),
                       'Data4LLMPrompting',
                       'ApplicationAPLMapping')


def _load_apl_mapping(app_name):
    """
    Load [label] -> [CSV column name] mapping from Data4LLMPrompting/ApplicationAPLMapping/<app_name>.txt
    Each line format: csv_colname,label
    """
    label_to_colnames = {}
    apl_map_file = join(APL_MAPPING_DIR, f'{app_name}.txt')

    try:
        with open(apl_map_file, 'r') as f_map:
            for line in f_map:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [x.strip() for x in line.split(',')]
                if len(parts) < 2:
                    continue
                colname, lbl = parts[0], parts[1]
                label_to_colnames.setdefault(lbl, []).append(colname)
    except FileNotFoundError:
        # No mapping file
        pass

    return label_to_colnames


def parse_kernel_info(kernel_info_file):
    """
    Parses kernel_info.txt lines.

    Returns:
        mapping: dict mapping CSV column name ->
                 - (label, loop_bound) for loops
                 - (label, dim_bounds, array_name) for arrays,
                   where dim_bounds is a dict: {dim_idx: bound, ...}
    """
    mapping = {}
    app_name = os.path.basename(os.path.dirname(kernel_info_file))

    # Load [label] -> [CSV colnames] mapping from ApplicationAPLMapping/<app_name>.txt
    label_to_colnames = _load_apl_mapping(app_name)
    has_apl_mapping = len(label_to_colnames) > 0

    try:
        with open(kernel_info_file, 'r') as f:
            # First non-empty line is usually the kernel function name
            first_line = None
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    first_line = line
                    break

            # Parse remaining lines
            for line in f:
                line = line.strip()
                if not line:
                    continue

                fields = [x.strip() for x in line.split(',')]
                if len(fields) < 3:
                    continue

                label = fields[0]
                kind = fields[1].lower()

                # LOOP entries: label, loop, bound
                if kind == 'loop':
                    try:
                        loop_bound = int(fields[2])
                    except ValueError:
                        continue

                    colnames = label_to_colnames.get(label)

                    if has_apl_mapping:
                        # only keep labels that appear in the APL mapping file
                        if not colnames:
                            continue
                        for cn in colnames:
                            mapping[cn] = (label, loop_bound)
                    else:
                        continue

                # ARRAY entries: label, array, array_name, [dim, bound, ...]
                elif kind == 'array':
                    array_name = fields[2]
                    dim_bounds = {}  # dim_idx -> bound

                    # starting at index 3: dim, bound, dim, bound, ...
                    i = 3
                    while i + 1 < len(fields):
                        dim_field = fields[i]
                        bound_field = fields[i + 1]
                        try:
                            dim_idx = int(dim_field)
                            dim_bound = int(bound_field)
                        except ValueError:
                            # stop parsing if we hit non-numeric tail
                            break
                        dim_bounds[dim_idx] = dim_bound
                        i += 2

                    if not dim_bounds:
                        continue

                    colnames = label_to_colnames.get(label)

                    if has_apl_mapping:
                        if not colnames:
                            continue
                        # (label, dim_bounds, array_name)
                        for cn in colnames:
                            mapping[cn] = (label, dim_bounds, array_name)
                    else:
                        continue

                else:
                    # Unknown kind, skip
                    continue

    except FileNotFoundError:
        return {}

    return mapping




class CSVResult:
    def __init__(self, point, perf, res_util, area, synth_time=None, weight=None, version=None, src_csv=None, row_idx=None):
        self.point = point            # dict with pragma keys ('_PIPE_*', '_UNROLL_*', '_ARRAY_T_*', '_ARRAY_F_*', '_ARRAY_D_*')
        self.perf = perf              # float latency (ms)
        self.res_util = res_util      # dict with keys 'util-BRAM', 'util-DSP', 'util-FF', 'util-LUT'
        self.synth_time = synth_time
        self.area = area              # "Area" from CSV --> (DSP_% + BRAM_% + FF_% + LUT_%) / 4.0
        self.weight = weight
        self.version = version
        self.src_csv = src_csv
        self.row_idx = row_idx


def load_csv_result_for_kernel(csv_file, kernel_info_map):
    """
    Read csv_file and return list of CSVResult.
    """
    results = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):  # idx=0 --> header
            # performance
            try:
                perf = float(row.get('Latency_msec', row.get('Latency', 0.0)))
            except Exception:
                perf = 0.0

            # area (already aggregated in CSV)
            try:
                area = float(row.get('Area', 0.0))
            except Exception:
                area = 0.0

            # resources utilization
            res_util = {}
            if row.get('BRAM_Utilization_percentage', '') != '':
                try:
                    res_util['util-BRAM'] = float(row['BRAM_Utilization_percentage']) / 100.0
                except Exception:
                    res_util['util-BRAM'] = 0.0
            if row.get('DSP_Utilization_percentage', '') != '':
                try:
                    res_util['util-DSP'] = float(row['DSP_Utilization_percentage']) / 100.0
                except Exception:
                    res_util['util-DSP'] = 0.0
            if row.get('FF_Utilization_percentage', '') != '':
                try:
                    res_util['util-FF'] = float(row['FF_Utilization_percentage']) / 100.0
                except Exception:
                    res_util['util-FF'] = 0.0
            if row.get('LUT_Utilization_percentage', '') != '':
                try:
                    res_util['util-LUT'] = float(row['LUT_Utilization_percentage']) / 100.0
                except Exception:
                    res_util['util-LUT'] = 0.0

            # pragma point
            point = {}
            for colname, info in kernel_info_map.items():
                if colname not in row:
                    continue

                # Loops: (label, loop_bound)
                # Arrays: (label, dim_bounds, array_name)
                if len(info) == 2:
                    label, aux = info            # aux = loop_bound (int)
                else:
                    label, aux, _array_name = info  # aux = dim_bounds (dict), not used by parse_token_to_point_cols for arrays

                token = row[colname]
                mapping = parse_token_to_point_cols(token, label, aux)
                point.update(mapping)

            # optional synthesis time
            synth_time = None
            if row.get('Synthesis_Time_sec', '') != '':
                try:
                    synth_time = float(row['Synthesis_Time_sec'])
                except Exception:
                    synth_time = None

            weight = None
            if row.get('Weight', '') != '':
                try:
                    weight = float(row['Weight'])
                except Exception:
                    weight = None

            version = row.get('Version', None)
            src_csv = basename(csv_file)
            row_idx = idx

            results.append(
                CSVResult(
                    point=point,
                    perf=perf,
                    area=area,
                    res_util=res_util,
                    synth_time=synth_time,
                    weight=weight,
                    version=version,
                    src_csv=src_csv,
                    row_idx=row_idx
                )
            )
    return results



def find_csv_for_kernel(csv_dir, kernel):
    """
    Find the CSV for a given kernel, accepting names like:
      preprocessed-<kernel>.csv
      preprocessed_<kernel>.csv
    and both '-' / '_' variants of <kernel> itself.

    Returns the full path or None if nothing is found.
    """
    candidates = set()

    # Base variants of the kernel name
    bases = {kernel, kernel.replace('-', '_'), kernel.replace('_', '-')}
    for base in bases:
        candidates.add(os.path.join(csv_dir, f'preprocessed-{base}.csv'))
        candidates.add(os.path.join(csv_dir, f'preprocessed_{base}.csv'))

    # Check the explicit candidates first
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback: fuzzy glob search (preprocessed-*<kernel>*.csv)
    pattern = os.path.join(csv_dir, f'preprocessed*{kernel}*.csv')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    return None



def parse_token_to_point_cols(token, label, bound):
    """
    Convert a CSV token into pipeline II and unroll FACTOR:
      _PIPE_<label> , _UNROLL_<label> , _ARRAY_T_<label>, _ARRAY_F_<label>, _ARRAY_D_<label>
    """
    pipe_key = f'_PIPE_{label}'
    unroll_key = f'_UNROLL_{label}'
    array_type = f'_ARRAY_T_{label}'
    array_factor = f'_ARRAY_F_{label}'
    array_dim = f'_ARRAY_D_{label}'

    token = (token or '').strip().lower()

    # default values
    pipe = 0
    unroll = 0

    # unroll (use bound when just 'unroll')
    if token == 'unroll':
        try:
            unroll = int(bound)
        except Exception:
            # if bound not convertible, keep 0 (same default as create_cpps)
            unroll = 0
        return {pipe_key: pipe, unroll_key: unroll}

    if token.startswith('unroll_'):
        parts = token.split('_', 1)
        try:
            unroll = int(parts[1])
        except Exception:
            unroll = 0
        return {pipe_key: pipe, unroll_key: unroll}

    # pipeline
    if token == 'pipeline':
        pipe = 1
        unroll = 0
        return {pipe_key: pipe, unroll_key: unroll}

    if token.startswith('pipeline_'):
        parts = token.split('_', 1)
        try:
            pipe = int(parts[1])
        except Exception:
            pipe = 1
        return {pipe_key: pipe, unroll_key: 0}

    if token.startswith('cyclic') or token.startswith('block'):
        parts = token.split('_')
        type = parts[0]
        factor = parts[1]
        dim = parts[2]
        return {array_type: type, array_factor: factor, array_dim: dim}

    if token.startswith('complete'):
        parts = token.split('_')
        dim = parts[1]
        return {array_type: 'complete', array_factor: 0, array_dim: dim}



def compute_global_max_pragma_length():
    csv_dir = join(get_root_path(), 'Data4LLMPrompting', 'preprocessed_CSVS')
    global_max = 0

    for kernel in ALL_KERNEL:
        csv_path = find_csv_for_kernel(csv_dir, kernel)
        if csv_path is None:
            continue

        kernel_info_path = join(get_root_path(), 'Data4LLMPrompting',
                                'ApplicationDataset', kernel, 'kernel_info.txt')
        kernel_info_map = parse_kernel_info(kernel_info_path)
        print(kernel_info_map)
        csv_result = load_csv_result_for_kernel(csv_path, kernel_info_map)
        if not csv_result:
            continue

        dim = len(csv_result[0].point)   # number of pragma slots for this kernel
        print(kernel, dim)
        global_max = max(global_max, dim)

    print("Global max pragma length =", global_max)
    return global_max

# compute_global_max_pragma_length()


TARGET = ['perf', 'area']
PRAGMAS = ['PIPELINE', 'UNROLL', 'ARRAY_PARTITION']

save_folder = '/home/ubuntu/save'
SAVE_DIR = join(save_folder, FLAGS.dataset, "all_kernels_2")

# ENCODER_PATH = join(SAVE_DIR, 'encoders')
ENCODER_PATH = join(get_root_path(), 'save', 'harp', 'all_kernels', 'encoders.klepto')
create_dir_if_not_exists(SAVE_DIR)


if FLAGS.dataset == 'harp':
    GEXF_FOLDER = join(get_root_path(), 'harp', 'processed', 'extended-pseudo-block-connected-hierarchy-2', '**')
else:
    raise NotImplementedError()


if FLAGS.all_kernels:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and FLAGS.graph_type in f])
else:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and f'{FLAGS.target_kernel}_' in f and FLAGS.graph_type in f])



def finte_diff_as_quality(new_result: Result, ref_result: Result) -> float:
    """Compute the quality of the point by finite difference method.

    Args:
        new_result: The new result to be qualified.
        ref_result: The reference result.

    Returns:
        The quality value (negative finite differnece). Larger the better.
    """

    def quantify_util(result: Result) -> float:
        """Quantify the resource utilization to a float number.

        util' = 5 * ceil(util / 5) for each util,
        area = sum(2^1(1/(1-util))) for each util'

        Args:
            result: The evaluation result.

        Returns:
            The quantified area value with the range (2*N) to infinite,
            where N is # of resources.
        """

        # Reduce the sensitivity to (100 / 5) = 20 intervals
        utils = [
            5 * ceil(u * 100 / 5) / 100 + FLAGS.epsilon for k, u in result.res_util.items()
            if k.startswith('util')
        ]

        # Compute the area
        return sum([2.0**(1.0 / (1.0 - u)) for u in utils])

    ref_util = quantify_util(ref_result)
    new_util = quantify_util(new_result)

    # if (new_result.perf / ref_result.perf) > 1.05:
    #     # Performance is too worse to be considered
    #     return -float('inf')

    if new_util == ref_util:
        if new_result.perf < ref_result.perf:
            # Free lunch
            # return float('inf')
            return FLAGS.max_number
        # Same util but slightly worse performance, neutral
        return 0

    return -(new_result.perf - ref_result.perf) / (new_util - ref_util)



class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None):
        super(MyOwnDataset, self).__init__(SAVE_DIR, transform, pre_transform)
        if data_files is not None:
            self.data_files = data_files

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if hasattr(self, 'data_files'):
            return self.data_files
        else:
            rtn = glob.glob(join(SAVE_DIR, '*.pt'))
            return rtn

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get_file_path(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        return fn

    def get(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        data = torch.load(fn, weights_only=False)
        return data

    def get(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        data = torch.load(fn, weights_only=False)

        if hasattr(data, "edge_id_to_idx"):
            del data.edge_id_to_idx

        return data


def split_dataset(dataset, train, val, dataset_test=None):
    file_li = dataset.processed_file_names
    li = random_split(file_li, [train, val, len(dataset) - train - val],
                          generator=torch.Generator().manual_seed(FLAGS.random_seed))
    if dataset_test is None:
        dataset_test = li[2]
    saver.log_info(f'{len(file_li)} graphs in total:'
          f' {len(li[0])} train {len(li[1])} val '
          f'{len(dataset_test)} test')
    train_dataset = MyOwnDataset(data_files=li[0])
    val_dataset = MyOwnDataset(data_files=li[1])
    test_dataset = MyOwnDataset(data_files=dataset_test)

    return [train_dataset, val_dataset, test_dataset]


def split_dataset_resample(dataset, train, val, test, test_id=0):
    file_li = dataset.processed_file_names
    num_batch = int(1 / test)
    splits_ratio = [int(len(dataset) * test)] * num_batch
    splits_ratio[-1] = len(dataset) - int(len(dataset) * test * (num_batch-1))
    # print(splits_ratio, len(dataset), sum(splits_ratio))
    splits_ = random_split(file_li, splits_ratio,
                          generator=torch.Generator().manual_seed(100))
    test_split = splits_[test_id]
    train_val_data = []
    for i in range(num_batch):
        if i != test_id:
            train_val_data.extend(splits_[i])
    new_train, new_val = int(len(train_val_data) * train / (train+val)), len(train_val_data) - int(len(train_val_data) * train / (train+val))
    li = random_split(train_val_data, [new_train, new_val],
                          generator=torch.Generator().manual_seed(100))
    saver.log_info(f'{len(file_li)} graphs in total:'
          f' {len(li[0])} train {len(li[1])} val '
          f'{len(test_split)} test')
    train_dataset = MyOwnDataset(data_files=li[0])
    val_dataset = MyOwnDataset(data_files=li[1])
    test_dataset = MyOwnDataset(data_files=test_split)
    return train_dataset, val_dataset, test_dataset


#def get_kernel_samples(dataset):
#    samples = defaultdict(list)
#    for data in dataset:
#        if f'{FLAGS.target_kernel}_' in data.gname:
#            samples[FLAGS.target_kernel].append(data)
#
#    return samples[FLAGS.target_kernel]


def get_kernel_samples(dataset):
    file_paths = []
    for idx in range(len(dataset)):
        g = dataset[idx]
        if g.gname == FLAGS.target_kernel:
            file_paths.append(dataset.get_file_path(idx))

    saver.log_info(f"Found {len(file_paths)} samples for kernel {FLAGS.target_kernel}")
    return MyOwnDataset(data_files=file_paths)




def split_train_test_kernel(dataset):
    samples = defaultdict(list)
    assert FLAGS.test_kernels is not None, 'No test_kernels selected'
    for idx, data in enumerate(dataset):
        if any(f'{kernel_name}_' in data.kernel for kernel_name in FLAGS.test_kernels):
            samples['test'].append(dataset.get_file_path(idx))
        else:
            samples['train'].append(dataset.get_file_path(idx))


    data_dict = defaultdict()
    data_dict['train'] = MyOwnDataset(data_files=samples['train'])
    # data_dict['test'] = MyOwnDataset(data_files=samples['test'])
    data_dict['test'] = samples['test']

    return data_dict


def log_graph_properties(ntypes, itypes, btypes, ftypes, ptypes, numerics):
    saver.log_info(f'\tntypes {len(ntypes)} {ntypes}')
    saver.log_info(f'\titypes {len(itypes)} {itypes}')
    saver.log_info(f'\tbtypes {len(btypes)} {btypes}')
    saver.log_info(f'\tftypes {len(ftypes)} {ftypes}')
    saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
    saver.log_info(f'\tnumerics {len(numerics)} {numerics}')





def _get_y(data, target):
    return getattr(data, target.replace('-', '_'))

def print_data_stats(data_loader, tvt):
    nns, ads, ys = [], [], []
    for d in tqdm(data_loader):
        nns.append(d.x.shape[0])
        # ads.append(d.edge_index.shape[1] / d.x.shape[0])
        ys.append(d.y.item())
    print_stats(nns, f'{tvt} number of nodes')
    # print_stats(ads, f'{tvt} avg degrees')
    plot_dist(ys, f'{tvt} ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
    saver.log_info(f'{tvt} ys', Counter(ys))


def load_encoders():
    rtn = load(ENCODER_PATH, saver.logdir)
    return rtn


def find_pragma_node(g, nid):
    pragma_nodes = {}
    for neighbor in g.neighbors(str(nid)):
        for pragma in ['pipeline', 'unroll', 'array_partition']:
            if g.nodes[neighbor]['text'].lower() == pragma:
                pragma_nodes[pragma] = neighbor
                break

    return pragma_nodes


def get_pragma_numeric(pragma_text, point, pragma_type):
    t_li = pragma_text.split(' ')
    pt = pragma_type.lower()

    if pt in ('pipeline', 'unroll'):
        numeric = 0
        for tok in t_li:
            if 'AUTO{' in tok.upper():
                # print(t_li[i])
                auto_what = _in_between(tok, '{', '}')
                val = point.get(auto_what, 0)
                if isinstance(val, int):
                    numeric = val
                else:
                    try:
                      numeric = int(val)
                    except:
                      numeric = 0

        return numeric

    elif pt == 'array_partition': ## array_partition
        partition_type = 0
        factor = 0
        dim = 0

        for tok in t_li:
            if 'AUTO{' in tok.upper():
                auto_what = _in_between(tok, '{', '}')
                val = point.get(auto_what, 0)
                low_tok = tok.lower()

                # type=auto{_ARRAY_T_*} --> 'cyclic'/'block'/'complete'
                if 'type=' in low_tok:
                    if not isinstance(val, int):
                        v = str(val).lower()
                        if v == 'cyclic':
                            partition_type = 100
                        elif v == 'block':
                            partition_type = 200
                        else:  # complete or anything else
                            partition_type = 300
                    else:
                        partition_type = val

                # factor=auto{_ARRAY_F_*} --> val should be int
                elif 'factor=' in low_tok:
                    if isinstance(val, int):
                        factor = val
                    else:
                        try:
                            factor = int(val)
                        except Exception:
                            factor = 0

                # dim=auto{_ARRAY_D_*} --> val should be int
                elif 'dim=' in low_tok:
                    if isinstance(val, int):
                        dim = val
                    else:
                        try:
                            dim = int(val)
                        except Exception:
                            dim = 0

        return partition_type, factor, dim

    # unknown pragma type
    return 0



def fill_pragma_vector(g, neighbor_pragmas, pragma_vector, point, node):
    '''
        # for each node, a vector of [pipeline II, unroll factor, array_partition type, array_partition factor, array_partition dim]
        # if no pragma assigned to node, a vector of [0, 0, 0, 0, 0]
    '''
    vector_id = {'pipeline': 0, 'unroll': 1, 'partition_type': 2, 'partition_factor': 3, 'partition_dim': 4}

    if len(pragma_vector):
        pragma_vector = [0, 0, 0, 0, 0]

    for pragma in ['pipeline', 'unroll', 'array_partition']:
        if pragma not in neighbor_pragmas:
            continue
        nid = neighbor_pragmas[pragma]
        pragma_text = g.nodes[nid]['full_text']

        if pragma in ('pipeline', 'unroll'):
            numeric = get_pragma_numeric(pragma_text, point, pragma_type=pragma)
            pragma_vector[vector_id[pragma]] = numeric
        else:  # array_partition
            partition_type, factor, dim = get_pragma_numeric(
                pragma_text, point, pragma_type='array_partition'
            )
            pragma_vector[vector_id['partition_type']] = partition_type
            pragma_vector[vector_id['partition_factor']] = factor
            pragma_vector[vector_id['partition_dim']] = dim

    # saver.log_info(f'point: {point}')
    # saver.log_info(f'{node}, {pragma_vector}')
    return pragma_vector



def encode_g_torch(g, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype):
    x_dict = _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, ftypes=None, btypes=None, point=None)
    X = _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
    edge_index = create_edge_index(g)

    return X, edge_index



def _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None,
                   itypes=None, ftypes=None, btypes=None, point=None):

    X_ntype = []      # node type <attribute id="3" title="type" type="long" />
    X_ptype = []      # pragma type (PIPELINE/UNROLL/ARRAY_PARTITION/NONE)
    X_numeric = []    # numeric scalar (used here only for ICMP)
    X_itype = []      # instruction type (text) <attribute id="2" title="text" type="string" />
    X_ftype = []      # function type <attribute id="1" title="function" type="long" />
    X_btype = []      # block type <attribute id="0" title="block" type="long" />

    X_contextnids = []      # 0/1: context node
    X_pragmanids = []       # 0/1: pragma node
    X_pseudonids = []       # 0/1: pseudo node
    X_icmpnids = []         # 0/1: icmp node

    # pragma-as-MLP:
    # per-node vector [pipeline II, unroll factor, array_partition type, array_partition factor, array_partition dim]
    X_pragma_per_node = []
    # 0/1: whether this pseudo node's pragma vector is nonzero
    X_pragmascopenids = []

    sorted_nodes = sorted(g.nodes(data=True), key=lambda x: int(x[0]))
    for nid, (node, ndata) in enumerate(sorted_nodes):
#    for nid, (node, ndata) in enumerate(g.nodes(data=True)):
        assert nid == int(node), f'{nid} {node}'

        # update global histograms (for fitting encoders)
        if ntypes is not None:
            ntypes[ndata['type']] += 1
        if itypes is not None:
            itypes[ndata['text']] += 1
        if btypes is not None:
            btypes[ndata['block']] += 1
        if ftypes is not None:
            ftypes[ndata['function']] += 1

        # pragma_as_MLP : 5D vector per pseudo node
        pragma_vector = [0, 0, 0, 0, 0]

        is_pseudo = isinstance(ndata.get('text', ''), str) and \
                    ('pseudo' in ndata['text'].lower())

        if is_pseudo:
            X_pseudonids.append(1)
            if FLAGS.pragma_scope == 'block':
                # find PIPELINE / UNROLL / ARRAY_PARTITION pragma nodes in this block
                neighbor_pragmas = find_pragma_node(g, node)
                if len(neighbor_pragmas) == 0:
                    X_pragmascopenids.append(0)
                else:
                    X_pragmascopenids.append(1)
                    pragma_vector = fill_pragma_vector(
                        g, neighbor_pragmas, pragma_vector, point, node
                    )
            else:
                raise NotImplementedError("Only pragma_scope = 'block' is supported.")
        else:
            X_pseudonids.append(0)
            X_pragmascopenids.append(0)

        X_pragma_per_node.append(pragma_vector)

        numeric = 0
        full_text = ndata.get('full_text', '')

        if isinstance(full_text, str) and 'icmp' in full_text:
            # e.g. "icmp ule i32 %i, 64" --> get the last token
            cmp_t = full_text.split(',')[-1].strip()
            if cmp_t.isdigit():
                cmp_val = int(cmp_t)
                numeric = cmp_val
                X_icmpnids.append(1)
            else:
                X_icmpnids.append(0)
        else:
            X_icmpnids.append(0)

        # pragma node / context node
        # ptype : PIPELINE / UNROLL / ARRAY_PARTITION / NONE
        if isinstance(full_text, str) and 'pragma' in full_text.lower():
            p_text = full_text.rstrip()
            assert p_text[:8].lower() == '#pragma ', f"Unexpected pragma format: {p_text}"
            p_body = p_text[8:]          # everything after '#pragma '
            p_body_up = p_body.upper()

            # mask variable name in ARRAY_PARTITION to avoid overfitting on specific array names in X_ptype
            if 'ARRAY_PARTITION' in p_body_up:
                t_li = p_body.split()
                for i, tok in enumerate(t_li):
                    if tok.upper().startswith('VARIABLE='):
                        t_li[i] = 'VARIABLE=<>'
                p_body = ' '.join(t_li)
                p_body_up = p_body.upper()

            if 'PIPELINE' in p_body_up:
                ptype = 'PIPELINE'
            elif 'UNROLL' in p_body_up:
                ptype = 'UNROLL'
            elif 'ARRAY_PARTITION' in p_body_up:
                ptype = 'ARRAY_PARTITION'
            else:
                ptype = 'NONE'

            X_pragmanids.append(1)
            X_contextnids.append(0)

        else:
            ptype = 'NONE'
            X_pragmanids.append(0)
            # context nodes = !(pseudo nodes) && !(pragma nodes)
            if is_pseudo:
                X_contextnids.append(0)
            else:
                X_contextnids.append(1)

        if ptypes is not None:
            ptypes[ptype] += 1
        if numerics is not None:
            numerics[numeric] += 1

        X_ntype.append([ndata['type']])
        X_ptype.append([ptype])
        X_numeric.append([numeric])
        X_itype.append([ndata['text']])
        X_ftype.append([ndata['function']])
        X_btype.append([ndata['block']])

    X_pragma_per_node = transform_X_torch(X_pragma_per_node)

    return {
        'X_ntype': X_ntype,
        'X_ptype': X_ptype,
        'X_numeric': X_numeric,
        'X_itype': X_itype,
        'X_ftype': X_ftype,
        'X_btype': X_btype,
        'X_contextnids': torch.FloatTensor(np.array(X_contextnids)),
        'X_pragmanids': torch.FloatTensor(np.array(X_pragmanids)),
        'X_pragmascopenids': torch.FloatTensor(np.array(X_pragmascopenids)),
        'X_pseudonids': torch.FloatTensor(np.array(X_pseudonids)),
        'X_icmpnids': torch.FloatTensor(np.array(X_icmpnids)),
        'X_pragma_per_node': X_pragma_per_node,
    }



def transform_X_torch(X):
    X = torch.FloatTensor(np.array(X))
    X = coo_matrix(X)
    X = _coo_to_sparse(X)
    X = X.to_dense()
    return X


def _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype):
    X_ntype = enc_ntype.transform(x_dict['X_ntype'])
    X_ptype = enc_ptype.transform(x_dict['X_ptype'])
    X_itype = enc_itype.transform(x_dict['X_itype'])
    X_ftype = enc_ftype.transform(x_dict['X_ftype'])
    X_btype = enc_btype.transform(x_dict['X_btype'])

    X_numeric = x_dict['X_numeric']
    X = hstack((X_ntype, X_ptype, X_numeric, X_itype, X_ftype, X_btype))
    X = _coo_to_sparse(X)
    X = X.to_dense()

    return X



def _encode_edge_dict(g, ftypes=None, ptypes=None):
    X_ftype = [] # flow type <attribute id="5" title="flow" type="long" />
    X_ptype = [] # position type <attribute id="6" title="position" type="long" />

    for nid1, nid2, edata in g.edges(data=True):
        X_ftype.append([edata['flow']])
        X_ptype.append([edata['position']])

    return {'X_ftype': X_ftype, 'X_ptype': X_ptype}


def _encode_edge_torch(edge_dict, enc_ftype, enc_ptype):
    X_ftype = enc_ftype.transform(edge_dict['X_ftype'])
    X_ptype = enc_ptype.transform(edge_dict['X_ptype'])

    if FLAGS.encode_edge_position:
        X = hstack((X_ftype, X_ptype))
    else:
        X = coo_matrix(X_ftype)
    if isinstance(X, csr_matrix):
        # Convert CSR to COO
        X = X.tocoo()
    X = _coo_to_sparse(X)
    X = X.to_dense()

    return X



def _in_between(text, left, right):
    return text[text.index(left) + len(left):text.index(right)]


def _check_any_in_str(li, s):
    for li_item in li:
        if li_item in s:
            return True
    return False


def create_edge_index(g):
#    g = nx.read_gexf(gexf_path, node_type=int)
#    edge_index = torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()
    g = nx.convert_node_labels_to_integers(g, ordering='sorted')
    edge_index = torch.LongTensor(list(g.edges)).t().contiguous()
    return edge_index


def _coo_to_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    rtn = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return rtn


def build_edge_id_to_idx(g):
    """
    Build a mapping from GEXF edge 'id' attribute (int) to
    the index position used in edge_index / edge_attr.
    Assumes we iterate edges in the same order as when we
    build edge_index and edge_attr.
    """
    edge_id_to_idx = {}
    for idx, (u, v, edata) in enumerate(g.edges(data=True)):
        # GEXF 'id' is usually stored as a string, convert to int
        edge_id = int(edata.get('id'))  # or just edata['id'] if you know it's there
        edge_id_to_idx[edge_id] = idx
    return edge_id_to_idx           


def get_data_list():
    """
    Two-pass, memory-friendly dataset builder.

    Pass 1 (vocab/meta):
      - Iterate over all GEXF graphs once.
      - Collect *unique* categorical tokens for OneHotEncoders (node/edge attrs).
      - Compute initial pragma dimensions per graph (init_feat_dict) and global max_pragma_length.
      - Do NOT build Data objects and do NOT keep per-row xy_dict in memory.

    Pass 2 (encoding + saving):
      - Iterate again over all GEXF + CSV.
      - For each CSV row, encode graph, build a Data object, and immediately save it to disk.
      - Accumulate only lightweight statistics (node counts, degrees, target distributions).
      - Never keep the full dataset in RAM (no giant data_list, no g.variants).
    """

    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')

    # For logging only (not used to fit encoders directly)
    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    # Encoders: either load from disk or create new ones
    if FLAGS.encoder_path is not None:
        saver.info(f'loading encoder from {FLAGS.encoder_path}')
        encoders = load(FLAGS.encoder_path, saver.logdir)
        enc_ntype = encoders['enc_ntype']
        enc_ptype = encoders['enc_ptype']
        enc_itype = encoders['enc_itype']
        enc_ftype = encoders['enc_ftype']
        enc_btype = encoders['enc_btype']

        enc_ftype_edge = encoders['enc_ftype_edge']  # 'flow'
        enc_ptype_edge = encoders['enc_ptype_edge']  # 'position'

        init_feat_dict = {}
        max_pragma_length = 93  # will be updated as we go in pass 2 if needed

    else:
        # New encoders – collect vocab in pass 1
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')

        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')

        init_feat_dict = {}
        max_pragma_length = 93

        # Unique tokens for fitting encoders (memory-light)
        ntype_tokens = set()
        ptype_tokens = set()
        itype_tokens = set()
        ftype_tokens = set()
        btype_tokens = set()
        edge_ftype_tokens = set()
        edge_ptype_tokens = set()

        # -------------------------
        # PASS 1: collect vocab/meta
        # -------------------------
        for gexf_file in tqdm(GEXF_FILES[0:]):
            saver.info(f'Collecting vocab from graph file: {gexf_file}')

            if FLAGS.dataset == 'harp':
                matched_kernel = None
                for k in ALL_KERNEL:
                    if f'{k}_' in gexf_file:
                        matched_kernel = k
                        break
                if matched_kernel is None:
                    saver.info('Skipping this file as the kernel name not in config.')
                    continue
                kernel = matched_kernel
            else:
                raise NotImplementedError()

            # Load graph
            g = nx.read_gexf(gexf_file)
            gname = os.path.basename(gexf_file).split('.')[0]

            # ---- Node vocab ----
            for nid, (node, ndata) in enumerate(g.nodes(data=True)):
                # Node-level categorical fields
                ntype_val = ndata['type']
                text_val = ndata['text']
                block_val = ndata['block']
                func_val = ndata['function']

                ntypes[ntype_val] += 1
                itypes[text_val] += 1
                btypes[block_val] += 1
                ftypes[func_val] += 1

                ntype_tokens.add(ntype_val)
                itype_tokens.add(text_val)
                btype_tokens.add(block_val)
                ftype_tokens.add(func_val)

                full_text = ndata.get('full_text', '')

                # ptype : PIPELINE / UNROLL / ARRAY_PARTITION / NONE
                if isinstance(full_text, str) and 'pragma' in full_text.lower():
                    p_text = full_text.rstrip()
                    # look after '#pragma ' if present
                    if p_text[:8].lower() == '#pragma ':
                        p_body = p_text[8:]
                    else:
                        p_body = p_text
                    p_body_up = p_body.upper()

                    if 'PIPELINE' in p_body_up:
                        ptype_val = 'PIPELINE'
                    elif 'UNROLL' in p_body_up:
                        ptype_val = 'UNROLL'
                    elif 'ARRAY_PARTITION' in p_body_up:
                        ptype_val = 'ARRAY_PARTITION'
                    else:
                        ptype_val = 'NONE'
                else:
                    ptype_val = 'NONE'

                ptypes[ptype_val] += 1
                ptype_tokens.add(ptype_val)

                # Numeric scalar distribution (icmp immediate), for logging only
                numeric_val = 0
                if isinstance(full_text, str) and 'icmp' in full_text:
                    cmp_t = full_text.split(',')[-1].strip()
                    if cmp_t.isdigit():
                        numeric_val = int(cmp_t)
                numerics[numeric_val] += 1

            # ---- Edge vocab ----
            for nid1, nid2, edata in g.edges(data=True):
                flow_val = edata['flow']
                pos_val = edata['position']

                ftypes_edge[flow_val] += 1
                ptypes_edge[pos_val] += 1

                edge_ftype_tokens.add(flow_val)
                edge_ptype_tokens.add(pos_val)

            # ---- CSV-based meta: pragma dim per graph ----
            csv_dir = join(get_root_path(), 'Data4LLMPrompting', 'preprocessed_CSVS')
            csv_path = find_csv_for_kernel(csv_dir, kernel)
            if csv_path is None:
                saver.warning(f'No CSV file found for kernel \"{kernel}\" in {csv_dir}. Skipping pragma meta.')
                del g
                gc.collect()
                continue

            kernel_info_path = join(get_root_path(), 'Data4LLMPrompting', 'ApplicationDataset', kernel, 'kernel_info.txt')
            kernel_info_map = parse_kernel_info(kernel_info_path)
            csv_result = load_csv_result_for_kernel(csv_path, kernel_info_map)

            if len(csv_result) == 0:
                saver.warning(f'No valid rows parsed for {kernel} (meta pass)')
                del g, csv_result
                gc.collect()
                continue

            first_pragmas_len = None
            for obj in csv_result:
                # Apply same regression filtering as later
                if FLAGS.task == 'regression' and (not FLAGS.invalid) and obj.perf < FLAGS.min_allowed_latency:
                    continue

                pragmas = []
                for name, value in sorted(obj.point.items()):
                    if ('_PIPE_' not in name and '_UNROLL_' not in name and '_ARRAY_' not in name):
                        continue

                    # Normalize pragma values to numeric (same logic as later)
                    if isinstance(value, str):
                        v = value.strip().lower()
                        if name.startswith('_ARRAY_T_'):
                            tmap = {'cyclic': 100, 'block': 200, 'complete': 300}
                            value = tmap.get(v, 0)
                        else:
                            try:
                                value = int(v)
                            except ValueError:
                                raise ValueError(f"Non-numeric pragma value '{value}' for key {name}")
                    elif not isinstance(value, int):
                        raise ValueError(f'Unexpected pragma value type: {type(value)}')

                    pragmas.append(value)

                first_pragmas_len = len(pragmas)
                break  # Only need first valid row

            if first_pragmas_len is not None:
                init_feat_dict[gname] = [first_pragmas_len]
                if first_pragmas_len > max_pragma_length:
                    max_pragma_length = first_pragmas_len

            saver.log_info(f'Graph {gname}: initial pragma dim {init_feat_dict.get(gname)}')

            del g, csv_result
            gc.collect()

        saver.log_info(f'Done vocab collection pass over {len(init_feat_dict)} graphs.')
        log_graph_properties(ntypes, itypes, btypes, ftypes, ptypes, numerics)

        # ---- Fit encoders on unique tokens (memory-light) ----
        if ntype_tokens:
            enc_ntype.fit([[t] for t in ntype_tokens])
        if ptype_tokens:
            enc_ptype.fit([[t] for t in ptype_tokens])
        if itype_tokens:
            enc_itype.fit([[t] for t in itype_tokens])
        if ftype_tokens:
            enc_ftype.fit([[t] for t in ftype_tokens])
        if btype_tokens:
            enc_btype.fit([[t] for t in btype_tokens])
        if edge_ftype_tokens:
            enc_ftype_edge.fit([[t] for t in edge_ftype_tokens])
        if edge_ptype_tokens:
            enc_ptype_edge.fit([[t] for t in edge_ptype_tokens])

        saver.log_info('Finished fitting OneHotEncoders.')

    # -----------------------------------
    # PASS 2: encode + save Data() to disk
    # -----------------------------------
    if FLAGS.force_regen:
        tmp_dir = SAVE_DIR + "_tmp"
        saver.log_info(f'Saving encoded graphs to disk at {tmp_dir}')

        # Clean temp dir
        if os.path.exists(tmp_dir):
            raise RuntimeError(
            f"Temporary dir {tmp_dir} already exists. "
            "This likely means a previous run died. "
            "Please inspect/backup it before rerunning."
            )
        create_dir_if_not_exists(tmp_dir)

        data_idx = 0
        tot_configs = 0
        num_files = 0

        # Light-weight stats (numbers only, no Data objects kept)
        nnodes_list = []
        degrees_list = []
        target_values = defaultdict(list)  # key: 'perf', 'area', 'actual_perf', ...

        for gexf_file in tqdm(GEXF_FILES[0:]):
            saver.info(f'Processing graph file (encoding): {gexf_file}')

            if FLAGS.dataset == 'harp':
                matched_kernel = None
                for k in ALL_KERNEL:
                    if f'{k}_' in gexf_file:
                        matched_kernel = k
                        break
                if matched_kernel is None:
                    saver.info('Skipping this file as the kernel name not in config.')
                    continue
                kernel = matched_kernel
            else:
                raise NotImplementedError()

            g = nx.read_gexf(gexf_file)
            gname = os.path.basename(gexf_file).split('.')[0]
            new_gname = gname.split('_')[0]

            csv_dir = join(get_root_path(), 'Data4LLMPrompting', 'preprocessed_CSVS')
            csv_path = find_csv_for_kernel(csv_dir, kernel)
            if csv_path is None:
                saver.warning(f'No CSV file found for kernel \"{kernel}\" in {csv_dir}. Skipping.')
                del g
                gc.collect()
                continue

            kernel_info_path = join(get_root_path(), 'Data4LLMPrompting', 'ApplicationDataset', kernel, 'kernel_info.txt')
            kernel_info_map = parse_kernel_info(kernel_info_path)
            csv_result = load_csv_result_for_kernel(csv_path, kernel_info_map)
            if len(csv_result) == 0:
                saver.warning(f'No valid rows parsed for {kernel} (encoding pass)')
                del g, csv_result
                gc.collect()
                continue

            # Reference design for kernel_speedup
            res_reference = None
            max_perf = 0.0
            for obj in csv_result:
                if obj.perf is None or obj.perf == 0:
                    continue
                if obj.perf > max_perf:
                    max_perf = obj.perf
                    res_reference = obj
            if res_reference is not None:
                saver.log_info(f'reference point for {kernel} is {res_reference.perf}')
            else:
                saver.log_info(f'did not find reference point for {kernel} with {len(csv_result)} points')

            edge_index = create_edge_index(g)
            edge_id_to_idx = build_edge_id_to_idx(g)

            cnt = 0
            for idx, obj in enumerate(csv_result):
                # Same filter logic as original
                if FLAGS.task == 'regression':
                    if not FLAGS.invalid and obj.perf < FLAGS.min_allowed_latency:
                        continue
                elif FLAGS.task == 'class':
                    pass
                else:
                    raise NotImplementedError()

                cnt += 1
                vname = f"csvrow_{idx}"

                # Encode node/edge categorical features for this CSV row
                xy_dict = _encode_X_dict(
                    g,
                    ntypes=None,
                    ptypes=None,
                    numerics=None,
                    itypes=None,
                    ftypes=None,
                    btypes=None,
                    point=obj.point,
                )
                edge_dict = _encode_edge_dict(g, ftypes=None, ptypes=None)

                # Build pragma vector (PIPELINE/UNROLL/ARRAY_PARTITION only)
                pragmas = []
                for name, value in sorted(obj.point.items()):
                    if ('_PIPE_' not in name and '_UNROLL_' not in name and '_ARRAY_' not in name):
                        continue

                    if isinstance(value, str):
                        v = value.strip().lower()
                        # _ARRAY_T_: type (cyclic/block/complete)
                        if name.startswith('_ARRAY_T_'):
                            tmap = {'cyclic': 100, 'block': 200, 'complete': 300}
                            value = tmap.get(v, 0)
                        else:
                            try:
                                value = int(v)
                            except ValueError:
                                raise ValueError(f"Non-numeric pragma value '{value}' for key {name}")
                    elif not isinstance(value, int):
                        raise ValueError(f'Unexpected pragma value type: {type(value)}')

                    pragmas.append(value)

                # Ensure consistent pragma dimension per graph
                check_dim = init_feat_dict.get(gname)
                if check_dim is not None:
                    assert check_dim[0] == len(pragmas), \
                        f'Pragma dim mismatch for {gname}: before {check_dim[0]}, now {len(pragmas)}'
                else:
                    init_feat_dict[gname] = [len(pragmas)]
                    if len(pragmas) > max_pragma_length:
                        max_pragma_length = len(pragmas)

                assert len(pragmas) <= max_pragma_length, \
                    f'Pragma length {len(pragmas)} exceeds max {max_pragma_length}'
                pragmas.extend([0] * (max_pragma_length - len(pragmas)))
                xy_dict['pragmas'] = torch.FloatTensor(np.array([pragmas]))

                # Targets (labels), same logic as original
                if FLAGS.task == 'regression':
                    for tname in TARGET:
                        if tname == 'perf':
                            perf_val = obj.perf if obj.perf is not None else 0.0

                            if FLAGS.norm_method == 'log2':
                                y = math.log2(perf_val + FLAGS.epsilon)
                            elif FLAGS.norm_method == 'const':
                                y = perf_val * FLAGS.normalizer
                            elif FLAGS.norm_method == 'off':
                                y = perf_val
                            elif 'speedup' in FLAGS.norm_method:
                                assert perf_val > 0.0
                                speedup = FLAGS.normalizer / perf_val
                                if FLAGS.norm_method == 'speedup-log2':
                                    y = math.log2(speedup + FLAGS.epsilon)
                                else:
                                    y = speedup
                            else:
                                raise NotImplementedError(
                                    f"Unsupported norm_method {FLAGS.norm_method} for perf"
                                )

                            xy_dict['perf'] = torch.FloatTensor(np.array([y]))
                            xy_dict['actual_perf'] = torch.FloatTensor(np.array([perf_val]))

                            if res_reference is not None and res_reference.perf not in (None, 0.0):
                                ks = math.log2(res_reference.perf / perf_val)
                            else:
                                ks = 0.0
                            xy_dict['kernel_speedup'] = torch.FloatTensor(np.array([ks]))

                        elif tname == 'area':
                            area_val = getattr(obj, 'area', 0.0)
                            if area_val <= 0.0:
                                area_val = FLAGS.epsilon

                            if FLAGS.norm_method == 'log2':
                                y = math.log2(area_val + FLAGS.epsilon)
                            elif FLAGS.norm_method == 'const':
                                y = area_val * FLAGS.util_normalizer
                            elif FLAGS.norm_method == 'off':
                                y = area_val
                            else:
                                y = math.log2(area_val + FLAGS.epsilon)

                            xy_dict['area'] = torch.FloatTensor(np.array([y]))
                            xy_dict['actual_area'] = torch.FloatTensor(np.array([area_val]))

                        elif 'util' in tname or 'total' in tname:
                            if tname not in obj.res_util:
                                y = 0.0
                            else:
                                y = obj.res_util[tname] * FLAGS.util_normalizer
                            xy_dict[tname] = torch.FloatTensor(np.array([y]))
                        else:
                            raise NotImplementedError(f"Unknown target name {tname}")

                elif FLAGS.task == 'class':
                    y = 0 if obj.perf < FLAGS.min_allowed_latency else 1
                    xy_dict['perf'] = (torch.FloatTensor(np.array([y]))
                                       .type(torch.LongTensor))
                else:
                    raise NotImplementedError()

                # Dense feature matrices using fitted encoders
                X = _encode_X_torch(xy_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
                edge_attr = _encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge)

                # Build Data object (not stored in a big list)
                if FLAGS.task == 'regression':
                    data_kwargs = dict(
                        gname=new_gname,
                        x=X,
                        key=vname,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        kernel=gname,
                        X_contextnids=xy_dict['X_contextnids'],
                        X_pragmanids=xy_dict['X_pragmanids'],
                        X_pragmascopenids=xy_dict['X_pragmascopenids'],
                        X_pseudonids=xy_dict['X_pseudonids'],
                        X_icmpnids=xy_dict['X_icmpnids'],
                        X_pragma_per_node=xy_dict['X_pragma_per_node'],
                        pragmas=xy_dict['pragmas'],
                        perf=xy_dict['perf'],
                        edge_id_to_idx=edge_id_to_idx,
                    )
                    if 'actual_perf' in xy_dict:
                        data_kwargs['actual_perf'] = xy_dict['actual_perf']
                    if 'kernel_speedup' in xy_dict:
                        data_kwargs['kernel_speedup'] = xy_dict['kernel_speedup']

                    for tname in TARGET:
                        if tname == 'perf':
                            continue
                        attr = tname.replace('-', '_')
                        if tname in xy_dict:
                            data_kwargs[attr] = xy_dict[tname]
                    if 'quality' in xy_dict:
                        data_kwargs['quality'] = xy_dict['quality']

                    data_obj = Data(**data_kwargs)
                    #data_obj = data_obj.sort()


                elif FLAGS.task == 'class':
                    data_obj = Data(
                        gname=new_gname,
                        x=X,
                        key=vname,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        kernel=gname,
                        X_contextnids=xy_dict['X_contextnids'],
                        X_pragmanids=xy_dict['X_pragmanids'],
                        X_pragmascopenids=xy_dict['X_pragmascopenids'],
                        X_pseudonids=xy_dict['X_pseudonids'],
                        X_icmpnids=xy_dict['X_icmpnids'],
                        X_pragma_per_node=xy_dict['X_pragma_per_node'],
                        pragmas=xy_dict['pragmas'],
                        perf=xy_dict['perf'],
                    )
                else:
                    raise NotImplementedError()

                # Lightweight stats
                nnodes_list.append(data_obj.x.shape[0])
                degrees_list.append(data_obj.edge_index.shape[1] / data_obj.x.shape[0])
                for target_name in ['perf', 'area', 'actual_perf']:
                    attr = target_name.replace('-', '_')
                    if hasattr(data_obj, attr):
                        target_values[target_name].append(
                            getattr(data_obj, attr).item()
                        )

                # Save immediately to disk (do not keep in memory)
                torch.save(data_obj, osp.join(tmp_dir, f'data_{data_idx}.pt'))
                data_idx += 1

            saver.log_info(f'final valid configs for {kernel}: {cnt}')
            tot_configs += cnt
            num_files += 1

            del g, csv_result, edge_index, edge_id_to_idx
            gc.collect()

        saver.log_info(f'Encoded {tot_configs} configurations across {num_files} graphs.')

        # Atomically replace SAVE_DIR with tmp_dir
        if os.path.exists(SAVE_DIR):
            rmtree(SAVE_DIR)
        os.rename(tmp_dir, SAVE_DIR)

        # Save encoders
        encoders_obj = {
            'enc_ntype': enc_ntype,
            'enc_ptype': enc_ptype,
            'enc_itype': enc_itype,
            'enc_ftype': enc_ftype,
            'enc_btype': enc_btype,
            'enc_ftype_edge': enc_ftype_edge,
            'enc_ptype_edge': enc_ptype_edge,
        }
        save(encoders_obj, ENCODER_PATH)

        # Save pragma dims
        for gname in init_feat_dict:
            init_feat_dict[gname].append(max_pragma_length)
        save(init_feat_dict, join(SAVE_DIR, 'pragma_dim'))
        for gname, feat_dim in init_feat_dict.items():
            saver.log_info(f'{gname} has initial dim {feat_dim[0]}')

        # Dataset-wide stats (using the numbers we collected)
        print_stats(nnodes_list, 'number of nodes')
        print_stats(degrees_list, 'avg degrees')
        TARGET.append('actual_perf')
        for target in TARGET:
            if target not in target_values or len(target_values[target]) == 0:
                saver.warning(f'Data does not have attribute {target} (for stats)')
                continue
            plot_dist(
                target_values[target],
                f'{target}_ys',
                saver.get_log_dir(),
                saver=saver,
                analyze_dist=True,
                bins=None
            )
            saver.log_info(f'{target}_ys', Counter(target_values[target]))

    # Final dataset loads .pt files from SAVE_DIR on demand
    rtn = MyOwnDataset()
    return rtn, init_feat_dict




###### Usage ######
# dataset, init_feat_dict = get_data_list()
# d0 = dataset[0]

# print("Graph structure")
# print("x:", d0.x[238])

# idx_73 = d0.edge_id_to_idx[73]
# print("edge_index:", d0.edge_index[:,idx_73])
# print("edge_attr:", d0.edge_attr[idx_73])

# print("Graph attributes")
# print("gname:", d0.gname)
# print("key:", d0.key)
# print("kernel:", d0.kernel)

# print("Node-level data")
# print("X_contextnids:", d0.X_contextnids)
# print("X_pragmanids:", d0.X_pragmanids)
# print("X_pseudonids:", d0.X_pseudonids)
