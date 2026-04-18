#-----------------------------------------------------------
#                         data.py
#-----------------------------------------------------------

import os
import glob
import torch
import gc
import math
import os.path as osp
import numpy as np
import networkx as nx
import csv

from glob import iglob
from torch_geometric.data import Dataset, Data
from torch.utils.data import random_split
from os.path import join
from math import ceil
from shutil import rmtree
from scipy.sparse import hstack, coo_matrix, csr_matrix
from collections import Counter, defaultdict, OrderedDict
from sklearn.preprocessing import OneHotEncoder
from os.path import join, basename

from result import Result
from utils import get_root_path, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, load, save
from config import FLAGS, ALL_KERNEL
from saver import saver
from tqdm import tqdm



APL_MAPPING_DIR = join(get_root_path(),
                       'Data4LLMPrompting',
                       'ApplicationAPLMapping')


def _find_apl_mapping_file(app_name):
    candidates = [
        join(APL_MAPPING_DIR, f'{app_name}.txt'),
        join(APL_MAPPING_DIR, f'{app_name.replace("-", "_")}.txt'),
        join(APL_MAPPING_DIR, f'{app_name.replace("_", "-")}.txt'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_apl_mapping(app_name):
    """
    Load [label] -> [CSV column name] mapping from
    Data4LLMPrompting/ApplicationAPLMapping/<app_name>.txt
    Each line format: csv_colname,label
    """
    label_to_colnames = {}

    apl_map_file = _find_apl_mapping_file(app_name)
    if apl_map_file is None:
        raise RuntimeError(
            f"No APL mapping file found for app '{app_name}' in {APL_MAPPING_DIR}. "
            f"Tried hyphen/underscore variants too."
        )

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

    if not label_to_colnames:
        raise RuntimeError(
            f"APL mapping file exists but is empty/unusable: {apl_map_file}"
        )

    return label_to_colnames


def parse_kernel_info(kernel_info_file):
    mapping = {}
    app_name = os.path.basename(os.path.dirname(kernel_info_file))

    label_to_colnames = _load_apl_mapping(app_name)

    try:
        with open(kernel_info_file, 'r') as f:
            # skip first non-empty line (kernel function name)
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    break

            for line in f:
                line = line.strip()
                if not line:
                    continue

                fields = [x.strip() for x in line.split(',')]
                if len(fields) < 3:
                    continue

                label = fields[0]
                kind = fields[1].lower()
                colnames = label_to_colnames.get(label, [])

                if not colnames:
                    continue

                if kind == 'loop':
                    try:
                        loop_bound = int(fields[2])
                    except ValueError:
                        continue

                    for cn in colnames:
                        mapping[cn] = (label, loop_bound)

                elif kind == 'array':
                    array_name = fields[2]
                    dim_bounds = {}

                    i = 3
                    while i + 1 < len(fields):
                        try:
                            dim_idx = int(fields[i])
                            dim_bound = int(fields[i + 1])
                        except ValueError:
                            break
                        dim_bounds[dim_idx] = dim_bound
                        i += 2

                    if not dim_bounds:
                        continue

                    for cn in colnames:
                        mapping[cn] = (label, dim_bounds, array_name)

    except FileNotFoundError:
        raise RuntimeError(f"kernel_info.txt not found: {kernel_info_file}")

    if not mapping:
        raise RuntimeError(
            f"parse_kernel_info() produced empty mapping for {kernel_info_file}. "
            f"This usually means the ApplicationAPLMapping file is missing, misnamed, "
            f"or does not match the labels in kernel_info.txt."
        )

    return mapping


def _as_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return default

PSEUDO_BLOCK_NODE_TYPE = 4
PRAGMA_NODE_TYPE = 100
ARRAY_SCOPE_NODE_TYPE = 104

def is_pseudo_block_node(ndata):
    return _as_int(ndata.get("type", -1), -1) == PSEUDO_BLOCK_NODE_TYPE

def is_array_scope_node(ndata):
    return _as_int(ndata.get("type", -1), -1) == ARRAY_SCOPE_NODE_TYPE

def is_pragma_node(ndata):
    return _as_int(ndata.get("type", -1), -1) == PRAGMA_NODE_TYPE


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
    pipe_key = f'_PIPE_{label}'
    unroll_key = f'_UNROLL_{label}'
    array_type = f'_ARRAY_T_{label}'
    array_factor = f'_ARRAY_F_{label}'
    array_dim = f'_ARRAY_D_{label}'

    token = (token or '').strip().lower()

    if token == '':
        return {}

    # -----------------
    # loop directives
    # -----------------
    if token == 'unroll':
        try:
            unroll = int(bound)
        except Exception:
            unroll = 0
        return {pipe_key: 0, unroll_key: unroll}

    if token.startswith('unroll_'):
        try:
            unroll = int(token.split('_', 1)[1])
        except Exception:
            unroll = 0
        return {pipe_key: 0, unroll_key: unroll}

    if token == 'pipeline':
        return {pipe_key: 1, unroll_key: 0}

    if token.startswith('pipeline_'):
        try:
            pipe = int(token.split('_', 1)[1])
        except Exception:
            pipe = 1
        return {pipe_key: pipe, unroll_key: 0}

    # -----------------
    # array directives
    # -----------------
    if token.startswith('cyclic_') or token.startswith('block_'):
        parts = token.split('_')
        if len(parts) != 3:
            raise ValueError(f"Malformed array token '{token}' for label {label}")
        return {
            array_type: parts[0],
            array_factor: int(parts[1]),
            array_dim: int(parts[2]),
        }

    if token.startswith('complete_'):
        parts = token.split('_')
        if len(parts) != 2:
            raise ValueError(f"Malformed array token '{token}' for label {label}")
        return {
            array_type: 'complete',
            array_factor: 0,
            array_dim: int(parts[1]),
        }

    raise ValueError(f"Unsupported pragma token '{token}' for label {label}")


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

save_folder = '/home/ubuntu/save'
SAVE_DIR = join(save_folder, FLAGS.dataset, "all_kernels_fixed")
GRAPH_DIR = join(SAVE_DIR, "graphs")
POINT_DIR = join(SAVE_DIR, "points")
INDEX_PATH = join(SAVE_DIR, "index.pt")
PRAGMA_DIM_PATH = join(SAVE_DIR, "pragma_dim")
ENCODER_PATH = join(SAVE_DIR, "encoders.klepto")
create_dir_if_not_exists(SAVE_DIR)


if FLAGS.dataset == 'harp':
    GEXF_FOLDER = join(get_root_path(), 'harp', 'processed', 'extended-pseudo-block-connected-hierarchy', '**')
else:
    raise NotImplementedError()


if FLAGS.all_kernels:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and FLAGS.graph_type in f])
else:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and f'{FLAGS.target_kernel}_' in f and FLAGS.graph_type in f])



def finite_diff_as_quality(new_result: Result, ref_result: Result) -> float:
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


def _check_finite_tensor(t, name, graph_name=None, local_idx=None, sanitize=False):
    if t is None or not torch.is_tensor(t):
        return t

    finite = torch.isfinite(t)
    if finite.all():
        return t

    bad = (~finite).sum().item()
    msg = f"Non-finite values in {name}"
    if graph_name is not None:
        msg += f" | graph={graph_name}"
    if local_idx is not None:
        msg += f" | local_idx={local_idx}"
    msg += f" | bad_count={bad}"

    if sanitize:
        print("[WARN]", msg, "-> applying torch.nan_to_num")
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    raise RuntimeError(msg)



class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None):
        # IMPORTANT: initialize records BEFORE calling parent constructor
        if data_files is not None:
            self.records = data_files
        else:
            self.records = torch.load(INDEX_PATH, weights_only=False)

        self._graph_cache = {}
        self._points_cache = {}

        super(MyOwnDataset, self).__init__(SAVE_DIR, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        # tell PyG that the processed artifacts live directly in SAVE_DIR
        return SAVE_DIR

    @property
    def processed_file_names(self):
        # PyG expects filenames, not dataset records
        return ['index.pt']

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.records)

    def __len__(self):
        return self.len()

    def get_file_path(self, idx):
        return self.records[idx]

    def _load_static_graph(self, graph_name):
        if graph_name not in self._graph_cache:
            obj = torch.load(join(GRAPH_DIR, f'{graph_name}.pt'), weights_only=False)

            obj['x'] = _check_finite_tensor(obj['x'].float(), 'x', graph_name=graph_name)
            obj['edge_attr'] = _check_finite_tensor(obj['edge_attr'].float(), 'edge_attr', graph_name=graph_name)

            for k in (
                'X_contextnids',
                'X_pragmanids',
                'X_pragmascopenids',
                'X_pseudonids',
                'X_arrayscopenids',
                'X_pipeline_scopeids',
                'X_unroll_scopeids',
                'X_array_partition_scopeids',
                'X_scopenids',
                'X_icmpnids',
            ):
                obj[k] = _check_finite_tensor(obj[k].float(), k, graph_name=graph_name)

            self._graph_cache[graph_name] = obj

        return self._graph_cache[graph_name]

    def _load_point_pack(self, graph_name):
        if graph_name not in self._points_cache:
            self._points_cache[graph_name] = torch.load(
                join(POINT_DIR, f'{graph_name}.pt'),
                weights_only=False
            )
        return self._points_cache[graph_name]

    def get(self, idx):
        rec = self.records[idx]
        graph_name = rec['graph_name']
        local_idx = rec['local_idx']

        g = self._load_static_graph(graph_name)
        p = self._load_point_pack(graph_name)

        kwargs = dict(
            gname=g['kernel_name'],
            graph_name=graph_name,
            kernel=g['kernel_name'],
            key=p['keys'][local_idx],

            x=g['x'],
            edge_index=g['edge_index'],
            edge_attr=g['edge_attr'],

            X_contextnids=g['X_contextnids'],
            X_pragmanids=g['X_pragmanids'],
            X_pragmascopenids=g['X_pragmascopenids'],
            X_pseudonids=g['X_pseudonids'],
            X_arrayscopenids=g['X_arrayscopenids'],
            X_pipeline_scopeids=g['X_pipeline_scopeids'],
            X_unroll_scopeids=g['X_unroll_scopeids'],
            X_array_partition_scopeids=g['X_array_partition_scopeids'],
            X_scopenids=g['X_scopenids'],
            X_icmpnids=g['X_icmpnids'],

            X_pragma_per_node=p['X_pragma_per_node'][local_idx].float(),
            pragmas=p['pragmas'][local_idx].float().unsqueeze(0),
        )

        if FLAGS.task == 'regression':
            kwargs.update(
                perf=p['perf'][local_idx].view(1).float(),
                actual_perf=p['actual_perf'][local_idx].view(1).float(),
                kernel_speedup=p['kernel_speedup'][local_idx].view(1).float(),
                area=p['area'][local_idx].view(1).float(),
                actual_area=p['actual_area'][local_idx].view(1).float(),
            )
        elif FLAGS.task == 'class':
            kwargs['perf'] = p['perf'][local_idx].view(1).long()
        else:
            raise NotImplementedError()
        
        kwargs['X_pragma_per_node'] = _check_finite_tensor(
            p['X_pragma_per_node'][local_idx].float(),
            'X_pragma_per_node',
            graph_name=graph_name,
            local_idx=local_idx,
        )

        kwargs['pragmas'] = _check_finite_tensor(
            p['pragmas'][local_idx].float().unsqueeze(0),
            'pragmas',
            graph_name=graph_name,
            local_idx=local_idx,
        )

        if FLAGS.task == 'regression':
            kwargs['perf'] = _check_finite_tensor(
                p['perf'][local_idx].view(1).float(),
                'perf',
                graph_name=graph_name,
                local_idx=local_idx,
            )
            kwargs['actual_perf'] = _check_finite_tensor(
                p['actual_perf'][local_idx].view(1).float(),
                'actual_perf',
                graph_name=graph_name,
                local_idx=local_idx,
            )
            kwargs['kernel_speedup'] = _check_finite_tensor(
                p['kernel_speedup'][local_idx].view(1).float(),
                'kernel_speedup',
                graph_name=graph_name,
                local_idx=local_idx,
            )
            kwargs['area'] = _check_finite_tensor(
                p['area'][local_idx].view(1).float(),
                'area',
                graph_name=graph_name,
                local_idx=local_idx,
            )
            kwargs['actual_area'] = _check_finite_tensor(
                p['actual_area'][local_idx].view(1).float(),
                'actual_area',
                graph_name=graph_name,
                local_idx=local_idx,
            )

        return Data(**kwargs)
    

def split_dataset(dataset, train, val, dataset_test=None):
    records = dataset.records

    splits = random_split(
        records,
        [train, val, len(records) - train - val],
        generator=torch.Generator().manual_seed(FLAGS.random_seed)
    )

    train_records = [splits[0][i] for i in range(len(splits[0]))]
    val_records   = [splits[1][i] for i in range(len(splits[1]))]

    if dataset_test is None:
        test_records = [splits[2][i] for i in range(len(splits[2]))]
    else:
        test_records = dataset_test

    saver.log_info(
        f'{len(records)} graphs in total:'
        f' {len(train_records)} train {len(val_records)} val '
        f'{len(test_records)} test'
    )

    train_dataset = MyOwnDataset(data_files=train_records)
    val_dataset   = MyOwnDataset(data_files=val_records)
    test_dataset  = MyOwnDataset(data_files=test_records)

    return [train_dataset, val_dataset, test_dataset]


def split_dataset_resample(dataset, train, val, test, test_id=0):
    records = dataset.records

    num_batch = int(1 / test)
    splits_ratio = [int(len(records) * test)] * num_batch
    splits_ratio[-1] = len(records) - int(len(records) * test * (num_batch - 1))

    splits_ = random_split(
        records,
        splits_ratio,
        generator=torch.Generator().manual_seed(100)
    )

    test_split = [splits_[test_id][i] for i in range(len(splits_[test_id]))]

    train_val_data = []
    for i in range(num_batch):
        if i != test_id:
            train_val_data.extend([splits_[i][j] for j in range(len(splits_[i]))])

    new_train = int(len(train_val_data) * train / (train + val))
    new_val = len(train_val_data) - new_train

    li = random_split(
        train_val_data,
        [new_train, new_val],
        generator=torch.Generator().manual_seed(100)
    )

    train_records = [li[0][i] for i in range(len(li[0]))]
    val_records   = [li[1][i] for i in range(len(li[1]))]

    saver.log_info(
        f'{len(records)} graphs in total:'
        f' {len(train_records)} train {len(val_records)} val '
        f'{len(test_split)} test'
    )

    train_dataset = MyOwnDataset(data_files=train_records)
    val_dataset   = MyOwnDataset(data_files=val_records)
    test_dataset  = MyOwnDataset(data_files=test_split)

    return train_dataset, val_dataset, test_dataset


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


def find_attached_pragmas(g, nid, allowed_kinds=None):
    nid = str(nid)
    pragma_nodes = {}

    center_data = g.nodes[nid]

    if is_array_scope_node(center_data):
        allowed_flows = {200}
    elif is_pseudo_block_node(center_data):
        allowed_flows = {4}
    else:
        return pragma_nodes

    def maybe_add(nb, edata):
        if _as_int(edata.get("flow", -1), -1) not in allowed_flows:
            return

        ndata = g.nodes[nb]
        if not is_pragma_node(ndata):
            return

        kind = str(ndata.get("text", "")).strip().lower()
        if allowed_kinds is not None and kind not in allowed_kinds:
            return

        prev = pragma_nodes.get(kind)
        if prev is not None and prev != nb:
            raise RuntimeError(
                f"Multiple attached pragma nodes of kind '{kind}' for scope node {nid}: {prev}, {nb}"
            )

        pragma_nodes[kind] = nb

    if g.is_multigraph():
        for _, nb, _, edata in g.out_edges(nid, keys=True, data=True):
            maybe_add(nb, edata)
        for nb, _, _, edata in g.in_edges(nid, keys=True, data=True):
            maybe_add(nb, edata)
    else:
        for _, nb, edata in g.out_edges(nid, data=True):
            maybe_add(nb, edata)
        for nb, _, edata in g.in_edges(nid, data=True):
            maybe_add(nb, edata)

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
    point = {} if point is None else point
    pragma_vector = [0, 0, 0, 0, 0]

    vector_id = {
        'pipeline': 0,
        'unroll': 1,
        'partition_type': 2,
        'partition_factor': 3,
        'partition_dim': 4,
    }

    for pragma in ('pipeline', 'unroll', 'array_partition'):
        if pragma not in neighbor_pragmas:
            continue

        pid = neighbor_pragmas[pragma]
        pragma_text = str(g.nodes[pid].get('full_text', ''))

        if pragma in ('pipeline', 'unroll'):
            numeric = get_pragma_numeric(pragma_text, point, pragma_type=pragma)
            pragma_vector[vector_id[pragma]] = numeric
        else:
            partition_type, factor, dim = get_pragma_numeric(
                pragma_text, point, pragma_type='array_partition'
            )
            pragma_vector[vector_id['partition_type']] = partition_type
            pragma_vector[vector_id['partition_factor']] = factor
            pragma_vector[vector_id['partition_dim']] = dim

    return pragma_vector


def encode_g_torch(g, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype):
    x_dict = _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, ftypes=None, btypes=None, point=None)
    X = _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
    edge_index = create_edge_index(g)

    return X, edge_index


def is_scope_anchor_node(ndata):
    return is_pseudo_block_node(ndata) or is_array_scope_node(ndata)


def _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None,
                   itypes=None, ftypes=None, btypes=None, point=None):

    X_ntype = []      # node type <attribute id="3" title="type" type="long" />
    X_ptype = []      # pragma type (PIPELINE/UNROLL/ARRAY_PARTITION/NONE)
    X_numeric = []    # numeric scalar (used here only for ICMP)
    X_itype = []      # instruction type (text) <attribute id="2" title="text" type="string" />
    X_ftype = []      # function type <attribute id="1" title="function" type="long" />
    X_btype = []      # block type <attribute id="0" title="block" type="long" />

    point = {} if point is None else point

    X_contextnids = []
    X_pragmanids = []
    X_pseudonids = []
    X_pipeline_scopeids = []
    X_unroll_scopeids = []
    X_array_partition_scopeids = []
    X_arrayscopenids = []
    X_scopenids = []
    X_icmpnids = []
    X_pragmascopenids = []
    X_pragma_per_node = []

    sorted_nodes = sorted(g.nodes(data=True), key=lambda x: int(x[0]))
    for nid, (node, ndata) in enumerate(sorted_nodes):
        assert nid == int(node), f'{nid} {node}'

        if ntypes is not None:
            ntypes[ndata['type']] += 1
        if itypes is not None:
            itypes[ndata['text']] += 1
        if btypes is not None:
            btypes[ndata['block']] += 1
        if ftypes is not None:
            ftypes[ndata['function']] += 1

        is_pseudo = is_pseudo_block_node(ndata)
        is_array_scope = is_array_scope_node(ndata)
        is_scope = is_pseudo or is_array_scope
        is_pragma = is_pragma_node(ndata)

        X_pseudonids.append(1 if is_pseudo else 0)
        X_arrayscopenids.append(1 if is_array_scope else 0)
        X_scopenids.append(1 if is_scope else 0)
        X_pragmanids.append(1 if is_pragma else 0)
        X_contextnids.append(0 if (is_scope or is_pragma) else 1)

        pragma_vector = [0, 0, 0, 0, 0]

        pipe_scope = 0
        unroll_scope = 0
        array_scope = 0

        if is_scope:
            if FLAGS.pragma_scope != 'block':
                raise NotImplementedError("Only pragma_scope='block' is supported.")

            allowed_kinds = {'pipeline', 'unroll'} if is_pseudo else {'array_partition'}
            neighbor_pragmas = find_attached_pragmas(g, node, allowed_kinds=allowed_kinds)

            if neighbor_pragmas:
                pragma_vector = fill_pragma_vector(g, neighbor_pragmas, pragma_vector, point, node)

                if is_pseudo:
                    pipe_scope = 1 if 'pipeline' in neighbor_pragmas else 0
                    unroll_scope = 1 if 'unroll' in neighbor_pragmas else 0
                elif is_array_scope:
                    array_scope = 1 if 'array_partition' in neighbor_pragmas else 0

        X_pipeline_scopeids.append(pipe_scope)
        X_unroll_scopeids.append(unroll_scope)
        X_array_partition_scopeids.append(array_scope)
        X_pragmascopenids.append(1 if (pipe_scope or unroll_scope or array_scope) else 0)

        X_pragma_per_node.append(pragma_vector)

        full_text = str(ndata.get('full_text', ''))
        numeric = 0
        if 'icmp' in full_text:
            cmp_t = full_text.split(',')[-1].strip()
            if cmp_t.isdigit():
                numeric = int(cmp_t)
                X_icmpnids.append(1)
            else:
                X_icmpnids.append(0)
        else:
            X_icmpnids.append(0)

        if is_pragma:
            kind_up = str(ndata.get('text', '')).strip().upper()
            if kind_up in ('PIPELINE', 'UNROLL', 'ARRAY_PARTITION'):
                ptype = kind_up
            else:
                ptype = 'NONE'
        else:
            ptype = 'NONE'

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
        'X_arrayscopenids': torch.FloatTensor(np.array(X_arrayscopenids)),
        'X_pipeline_scopeids': torch.FloatTensor(np.array(X_pipeline_scopeids)),
        'X_unroll_scopeids': torch.FloatTensor(np.array(X_unroll_scopeids)),
        'X_array_partition_scopeids': torch.FloatTensor(np.array(X_array_partition_scopeids)),
        'X_scopenids': torch.FloatTensor(np.array(X_scopenids)),
        'X_icmpnids': torch.FloatTensor(np.array(X_icmpnids)),
        'X_pragma_per_node': transform_X_torch(X_pragma_per_node),
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


def _mask_tensor(x):
    return torch.tensor(np.array(x), dtype=torch.bool)

def _float16_tensor(x):
    return torch.tensor(np.array(x), dtype=torch.float16)

def build_dynamic_pragma_per_node(g, point):
    """
    Dynamic per-design tensor: [num_nodes, 5]
    Columns:
      0 pipeline
      1 unroll
      2 partition_type
      3 partition_factor
      4 partition_dim
    Stored compactly as int16 on disk.
    """
    point = {} if point is None else point
    rows = []

    sorted_nodes = sorted(g.nodes(data=True), key=lambda x: int(x[0]))
    for nid, (node, ndata) in enumerate(sorted_nodes):
        assert nid == int(node), f'{nid} {node}'

        pragma_vector = [0, 0, 0, 0, 0]

        is_pseudo = is_pseudo_block_node(ndata)
        is_array_scope = is_array_scope_node(ndata)
        is_scope = is_pseudo or is_array_scope

        if is_scope:
            allowed_kinds = {'pipeline', 'unroll'} if is_pseudo else {'array_partition'}
            neighbor_pragmas = find_attached_pragmas(g, node, allowed_kinds=allowed_kinds)
            if neighbor_pragmas:
                pragma_vector = fill_pragma_vector(g, neighbor_pragmas, pragma_vector, point, node)

        rows.append(pragma_vector)

    return torch.tensor(rows, dtype=torch.int16)


def build_static_graph_payload(g, graph_name, kernel_name,
                               enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype,
                               enc_ftype_edge, enc_ptype_edge):
    """
    Static graph data, saved once per graph.
    """
    x_dict = _encode_X_dict(
        g,
        ntypes=None, ptypes=None, numerics=None,
        itypes=None, ftypes=None, btypes=None,
        point=None,   # IMPORTANT: static pass, no design-point values
    )

    X = _encode_X_torch(
        x_dict,
        enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype
    ).to(torch.float32).contiguous()

    edge_index = create_edge_index(g).contiguous()

    edge_dict = _encode_edge_dict(g, ftypes=None, ptypes=None)
    edge_attr = _encode_edge_torch(
        edge_dict, enc_ftype_edge, enc_ptype_edge
    ).to(torch.float16).contiguous()

    if not torch.isfinite(X).all():
        bad = (~torch.isfinite(X)).sum().item()
        raise RuntimeError(f"Non-finite node features before save for {graph_name}: bad_count={bad}")

    if not torch.isfinite(edge_attr.float()).all():
        bad = (~torch.isfinite(edge_attr.float())).sum().item()
        raise RuntimeError(f"Non-finite edge_attr before save for {graph_name}: bad_count={bad}")

    return {
        'graph_name': graph_name,
        'kernel_name': kernel_name,
        'x': X,
        'edge_index': edge_index,
        'edge_attr': edge_attr,

        # static masks: store compactly as bool
        'X_contextnids': x_dict['X_contextnids'].bool(),
        'X_pragmanids': x_dict['X_pragmanids'].bool(),
        'X_pragmascopenids': x_dict['X_pragmascopenids'].bool(),
        'X_pseudonids': x_dict['X_pseudonids'].bool(),
        'X_arrayscopenids': x_dict['X_arrayscopenids'].bool(),
        'X_pipeline_scopeids': x_dict['X_pipeline_scopeids'].bool(),
        'X_unroll_scopeids': x_dict['X_unroll_scopeids'].bool(),
        'X_array_partition_scopeids': x_dict['X_array_partition_scopeids'].bool(),
        'X_scopenids': x_dict['X_scopenids'].bool(),
        'X_icmpnids': x_dict['X_icmpnids'].bool(),
    }


def _in_between(text, left, right):
    return text[text.index(left) + len(left):text.index(right)]


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


def get_ptype_from_node(ndata):
    if not is_pragma_node(ndata):
        return 'NONE'

    kind_up = str(ndata.get('text', '')).strip().upper()
    if kind_up in ('PIPELINE', 'UNROLL', 'ARRAY_PARTITION'):
        return kind_up
    return 'NONE'


def build_pragmas_list_from_point(point):
    pragmas = []
    for name, value in sorted(point.items()):
        if not name.startswith(('_PIPE_', '_UNROLL_', '_ARRAY_T_', '_ARRAY_F_', '_ARRAY_D_')):
            continue

        if isinstance(value, str):
            v = value.strip().lower()
            if name.startswith('_ARRAY_T_'):
                value = {'cyclic': 100, 'block': 200, 'complete': 300}.get(v, 0)
            else:
                value = int(v)
        elif not isinstance(value, int):
            raise ValueError(f'Unexpected pragma value type: {type(value)} for key {name}')

        pragmas.append(value)
    return pragmas


def get_data_list():
    """
    Build the fixed all-kernel dataset for GNN regression/classification.

    Design:
      1) Always run a metadata pass:
         - collect per-graph pragma dimensionality
         - compute global max_pragma_length
         - collect vocab stats
      2) Fit encoders only if FLAGS.encoder_path is None
      3) If FLAGS.force_regen is True, encode and save one .pt per design point

    Returns:
        dataset: MyOwnDataset
        init_feat_dict: dict[graph_name] = [initial_pragma_dim, max_pragma_length]
    """
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')

    # -----------------------------
    # helpers
    # -----------------------------
    def _resolve_kernel_from_gexf(gexf_file):
        if FLAGS.dataset != 'harp':
            raise NotImplementedError()

        for k in ALL_KERNEL:
            if f'{k}_' in gexf_file:
                return k
        return None

    def _load_graph_and_csv(gexf_file, kernel):
        g = nx.read_gexf(gexf_file)

        csv_dir = join(get_root_path(), 'Data4LLMPrompting', 'preprocessed_CSVS')
        csv_path = find_csv_for_kernel(csv_dir, kernel)
        if csv_path is None:
            return g, None, None

        kernel_info_path = join(
            get_root_path(),
            'Data4LLMPrompting',
            'ApplicationDataset',
            kernel,
            'kernel_info.txt'
        )
        kernel_info_map = parse_kernel_info(kernel_info_path)
        if not kernel_info_map:
            raise RuntimeError(
                f"Empty kernel_info_map for kernel '{kernel}'. "
                f"Check ApplicationAPLMapping and kernel_info.txt."
            )
        csv_result = load_csv_result_for_kernel(csv_path, kernel_info_map)
        if csv_result and all(len(obj.point) == 0 for obj in csv_result):
            raise RuntimeError(
                f"All CSV rows for kernel '{kernel}' produced empty pragma points. "
                f"This means the CSV headers were not mapped to kernel_info labels."
            )
        return g, csv_path, csv_result

    def _should_keep_obj(obj):
        if FLAGS.task == 'regression':
            if (not FLAGS.invalid) and obj.perf < FLAGS.min_allowed_latency:
                return False
            return True
        elif FLAGS.task == 'class':
            return True
        else:
            raise NotImplementedError()

    def _fit_encoder(enc, token_set):
        if token_set:
            enc.fit([[t] for t in token_set])

    # -----------------------------
    # pass-1 bookkeeping
    # -----------------------------
    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    fit_new_encoders = (FLAGS.encoder_path is None)

    if fit_new_encoders:
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')

        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')

        ntype_tokens = set()
        ptype_tokens = set()
        itype_tokens = set()
        ftype_tokens = set()
        btype_tokens = set()
        edge_ftype_tokens = set()
        edge_ptype_tokens = set()
    else:
        saver.info(f'loading encoder from {FLAGS.encoder_path}')
        encoders = load(FLAGS.encoder_path, saver.logdir)
        enc_ntype = encoders['enc_ntype']
        enc_ptype = encoders['enc_ptype']
        enc_itype = encoders['enc_itype']
        enc_ftype = encoders['enc_ftype']
        enc_btype = encoders['enc_btype']
        enc_ftype_edge = encoders['enc_ftype_edge']
        enc_ptype_edge = encoders['enc_ptype_edge']

        # placeholders so later code stays simple
        ntype_tokens = None
        ptype_tokens = None
        itype_tokens = None
        ftype_tokens = None
        btype_tokens = None
        edge_ftype_tokens = None
        edge_ptype_tokens = None

    init_feat_dict = {}
    max_pragma_length = 0

    saver.log_info('Starting metadata/vocab pass...')
    for gexf_file in tqdm(GEXF_FILES):
        kernel = _resolve_kernel_from_gexf(gexf_file)
        if kernel is None:
            saver.info(f'Skipping file not matched to kernel list: {gexf_file}')
            continue

        graph_name = os.path.basename(gexf_file).split('.')[0]
        g, csv_path, csv_result = _load_graph_and_csv(gexf_file, kernel)

        # ----- node / edge vocab and stats -----
        for _, ndata in g.nodes(data=True):
            ntype_val = ndata['type']
            text_val = ndata['text']
            block_val = ndata['block']
            func_val = ndata['function']
            ptype_val = get_ptype_from_node(ndata)

            ntypes[ntype_val] += 1
            itypes[text_val] += 1
            btypes[block_val] += 1
            ftypes[func_val] += 1
            ptypes[ptype_val] += 1

            full_text = ndata.get('full_text', '')
            numeric_val = 0
            if isinstance(full_text, str) and 'icmp' in full_text:
                cmp_t = full_text.split(',')[-1].strip()
                if cmp_t.isdigit():
                    numeric_val = int(cmp_t)
            numerics[numeric_val] += 1

            if fit_new_encoders:
                ntype_tokens.add(ntype_val)
                itype_tokens.add(text_val)
                btype_tokens.add(block_val)
                ftype_tokens.add(func_val)
                ptype_tokens.add(ptype_val)

        for _, _, edata in g.edges(data=True):
            flow_val = edata['flow']
            pos_val = edata['position']

            ftypes_edge[flow_val] += 1
            ptypes_edge[pos_val] += 1

            if fit_new_encoders:
                edge_ftype_tokens.add(flow_val)
                edge_ptype_tokens.add(pos_val)

        # ----- pragma-dim metadata -----
        if csv_path is None:
            saver.warning(f'No CSV file found for kernel "{kernel}". Skipping pragma meta for {graph_name}.')
            del g
            gc.collect()
            continue

        if not csv_result:
            saver.warning(f'No valid rows parsed for {kernel} (meta pass).')
            del g, csv_result
            gc.collect()
            continue

        first_pragmas_len = None
        for obj in csv_result:
            if not _should_keep_obj(obj):
                continue
            pragmas = build_pragmas_list_from_point(obj.point)
            first_pragmas_len = len(pragmas)
            break

        if first_pragmas_len is not None:
            init_feat_dict[graph_name] = [first_pragmas_len]
            max_pragma_length = max(max_pragma_length, first_pragmas_len)
        else:
            saver.warning(f'No kept rows after filtering for graph {graph_name}.')

        saver.log_info(f'Graph {graph_name}: initial pragma dim {init_feat_dict.get(graph_name)}')

        del g, csv_result
        gc.collect()

    saver.log_info(f'Done metadata pass over {len(init_feat_dict)} graphs.')
    log_graph_properties(ntypes, itypes, btypes, ftypes, ptypes, numerics)

    if max_pragma_length <= 0:
        raise RuntimeError(
            "max_pragma_length stayed 0. "
            "This means no valid CSV design points survived filtering."
        )

    # -----------------------------
    # fit encoders only if needed
    # -----------------------------
    if fit_new_encoders:
        _fit_encoder(enc_ntype, ntype_tokens)
        _fit_encoder(enc_ptype, ptype_tokens)
        _fit_encoder(enc_itype, itype_tokens)
        _fit_encoder(enc_ftype, ftype_tokens)
        _fit_encoder(enc_btype, btype_tokens)
        _fit_encoder(enc_ftype_edge, edge_ftype_tokens)
        _fit_encoder(enc_ptype_edge, edge_ptype_tokens)
        saver.log_info('Finished fitting OneHotEncoders.')

    # -----------------------------
    # pass-2 encode + save
    # -----------------------------
    if FLAGS.force_regen:
        tmp_dir = SAVE_DIR + "_tmp"
        graph_tmp_dir = join(tmp_dir, "graphs")
        point_tmp_dir = join(tmp_dir, "points")

        saver.log_info(f'Saving compact encoded dataset to {tmp_dir}')

        if os.path.exists(tmp_dir):
            raise RuntimeError(
                f"Temporary dir {tmp_dir} already exists. "
                "A previous run likely died. Inspect/remove it first."
            )

        create_dir_if_not_exists(tmp_dir)
        create_dir_if_not_exists(graph_tmp_dir)
        create_dir_if_not_exists(point_tmp_dir)

        global_index = []
        tot_configs = 0
        num_files = 0

        nnodes_list = []
        degrees_list = []
        target_values = defaultdict(list)

        saver.log_info('Starting compact encoding/saving pass...')
        for gexf_file in tqdm(GEXF_FILES):
            kernel = _resolve_kernel_from_gexf(gexf_file)
            if kernel is None:
                saver.info(f'Skipping file not matched to kernel list: {gexf_file}')
                continue

            graph_name = os.path.basename(gexf_file).split('.')[0]
            kernel_name = kernel

            g, csv_path, csv_result = _load_graph_and_csv(gexf_file, kernel)
            if csv_path is None:
                saver.warning(f'No CSV file found for kernel "{kernel}". Skipping {graph_name}.')
                del g
                gc.collect()
                continue

            if not csv_result:
                saver.warning(f'No valid rows parsed for {kernel} (encoding pass).')
                del g, csv_result
                gc.collect()
                continue

            # reference point for kernel_speedup
            res_reference = None
            max_perf = 0.0
            for obj in csv_result:
                if obj.perf is None or obj.perf == 0:
                    continue
                if obj.perf > max_perf:
                    max_perf = obj.perf
                    res_reference = obj

            # save static graph ONCE
            static_payload = build_static_graph_payload(
                g, graph_name, kernel_name,
                enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype,
                enc_ftype_edge, enc_ptype_edge
            )
            torch.save(static_payload, join(graph_tmp_dir, f'{graph_name}.pt'))

            nnodes_list.append(static_payload['x'].shape[0])
            degrees_list.append(static_payload['edge_index'].shape[1] / static_payload['x'].shape[0])

            # accumulate all point-wise tensors for this graph
            keys = []
            pragmas_list = []
            pragma_per_node_list = []

            perf_list = []
            actual_perf_list = []
            kernel_speedup_list = []
            area_list = []
            actual_area_list = []

            local_idx = 0

            for row_idx, obj in enumerate(csv_result):
                if not _should_keep_obj(obj):
                    continue

                key_name = f"csvrow_{row_idx}"

                pragmas = build_pragmas_list_from_point(obj.point)
                check_dim = init_feat_dict.get(graph_name)
                if check_dim is None:
                    raise RuntimeError(f'Graph {graph_name} missing from init_feat_dict.')

                if check_dim[0] != len(pragmas):
                    raise RuntimeError(
                        f'Pragma dim mismatch for {graph_name}: '
                        f'meta pass={check_dim[0]}, current={len(pragmas)}'
                    )

                if len(pragmas) > max_pragma_length:
                    raise RuntimeError(
                        f'Pragma length {len(pragmas)} exceeds max_pragma_length {max_pragma_length}'
                    )

                pragmas = pragmas + [0] * (max_pragma_length - len(pragmas))

                # dynamic per-node pragma values
                pragma_per_node = build_dynamic_pragma_per_node(g, obj.point)

                # targets
                if FLAGS.task == 'regression':
                    perf_val = obj.perf if obj.perf is not None else 0.0
                    area_val = obj.area if obj.area is not None else 0.0
                    area_safe = area_val if area_val > 0.0 else FLAGS.epsilon

                    if FLAGS.norm_method == 'log2':
                        perf_y = math.log2(perf_val + FLAGS.epsilon)
                    elif FLAGS.norm_method == 'const':
                        perf_y = perf_val * FLAGS.normalizer
                    elif FLAGS.norm_method == 'off':
                        perf_y = perf_val
                    elif 'speedup' in FLAGS.norm_method:
                        if perf_val <= 0.0:
                            perf_y = 0.0
                        else:
                            speedup = FLAGS.normalizer / perf_val
                            if FLAGS.norm_method == 'speedup-log2':
                                perf_y = math.log2(speedup + FLAGS.epsilon)
                            else:
                                perf_y = speedup
                    else:
                        raise NotImplementedError(f"Unsupported norm_method {FLAGS.norm_method} for perf")

                    if FLAGS.norm_method == 'const':
                        area_y = area_safe * FLAGS.util_normalizer
                    elif FLAGS.norm_method == 'off':
                        area_y = area_safe
                    else:
                        area_y = math.log2(area_safe + FLAGS.epsilon)

                    if res_reference is not None and res_reference.perf not in (None, 0.0) and perf_val > 0.0:
                        ks = math.log2(res_reference.perf / perf_val)
                    else:
                        ks = 0.0

                    perf_list.append(perf_y)
                    actual_perf_list.append(perf_val)
                    kernel_speedup_list.append(ks)
                    area_list.append(area_y)
                    actual_area_list.append(area_val)

                    target_values['perf'].append(perf_y)
                    target_values['actual_perf'].append(perf_val)
                    target_values['area'].append(area_y)
                    target_values['actual_area'].append(area_val)

                elif FLAGS.task == 'class':
                    cls_y = 0 if obj.perf < FLAGS.min_allowed_latency else 1
                    perf_list.append(cls_y)
                else:
                    raise NotImplementedError()

                keys.append(key_name)
                pragmas_list.append(torch.tensor(pragmas, dtype=torch.int16))
                pragma_per_node_list.append(pragma_per_node)

                global_index.append({
                    'graph_name': graph_name,
                    'local_idx': local_idx,
                })
                local_idx += 1

            if local_idx == 0:
                saver.warning(f'No kept rows for {graph_name} in encoding pass.')
                del g, csv_result
                gc.collect()
                continue

            point_payload = {
                'graph_name': graph_name,
                'kernel_name': kernel_name,
                'keys': keys,
                'pragmas': torch.stack(pragmas_list, dim=0),                  # [N, P], int16
                'X_pragma_per_node': torch.stack(pragma_per_node_list, dim=0) # [N, num_nodes, 5], int16
            }

            if FLAGS.task == 'regression':
                point_payload.update({
                    'perf': torch.tensor(perf_list, dtype=torch.float32),
                    'actual_perf': torch.tensor(actual_perf_list, dtype=torch.float32),
                    'kernel_speedup': torch.tensor(kernel_speedup_list, dtype=torch.float32),
                    'area': torch.tensor(area_list, dtype=torch.float32),
                    'actual_area': torch.tensor(actual_area_list, dtype=torch.float32),
                })
            else:
                point_payload['perf'] = torch.tensor(perf_list, dtype=torch.long)

            torch.save(point_payload, join(point_tmp_dir, f'{graph_name}.pt'))

            saver.log_info(f'final valid configs for {kernel}: {local_idx}')
            tot_configs += local_idx
            num_files += 1

            del g, csv_result, static_payload, point_payload
            gc.collect()

        saver.log_info(f'Encoded {tot_configs} configurations across {num_files} graphs.')

        torch.save(global_index, join(tmp_dir, 'index.pt'))

        encoders_obj = {
            'enc_ntype': enc_ntype,
            'enc_ptype': enc_ptype,
            'enc_itype': enc_itype,
            'enc_ftype': enc_ftype,
            'enc_btype': enc_btype,
            'enc_ftype_edge': enc_ftype_edge,
            'enc_ptype_edge': enc_ptype_edge,
        }
        save(encoders_obj, join(tmp_dir, 'encoders.klepto'))

        pragma_dim_to_save = {}
        for graph_name, feat_dim in init_feat_dict.items():
            pragma_dim_to_save[graph_name] = [feat_dim[0], max_pragma_length]
        save(pragma_dim_to_save, join(tmp_dir, 'pragma_dim'))

        if os.path.exists(SAVE_DIR):
            rmtree(SAVE_DIR)
        os.rename(tmp_dir, SAVE_DIR)

        print_stats(nnodes_list, 'number of nodes')
        print_stats(degrees_list, 'avg degrees')

        stats_targets = list(TARGET) + ['actual_perf', 'actual_area']
        for target in stats_targets:
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

    dataset = MyOwnDataset()
    return dataset, init_feat_dict


########## Run ##########
if __name__ == "__main__":
    dataset, init_feat_dict = get_data_list()
    print(f"Built dataset with {len(dataset)} samples")
    print(f"Number of graphs in pragma_dim: {len(init_feat_dict)}")

