#-----------------------------------------------------------
#                       config.py
#-----------------------------------------------------------

from utils import get_user, get_host, get_root_path
import argparse
import torch
from glob import iglob
from os.path import join
import argparse
import torch
from glob import iglob
from os.path import join

decoder_arch = []

parser = argparse.ArgumentParser()
# TASK = 'class'
TASK = 'regression'
parser.add_argument('--task', default=TASK)

# SUBTASK = 'dse'
# SUBTASK = 'inference'
SUBTASK = 'train'
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--plot_dse', default=False)


#################### visualization ####################
parser.add_argument('--vis_per_kernel', default=True) ## only tsne visualization for now


######################## data ########################

TARGETS = ['perf', 'area'] #, 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF']

ALL_KERNEL = ['machsuite-gemm-blocked', 'machsuite-gemm-ncubed', 'machsuite-md-knn', 'machsuite-sort-radix',
              'machsuite-spmv-ellpack', 'machsuite-stencil2d', 'machsuite-stencil3d', 'machsuite-viterbi',
              'rodinia-backprop-0-baseline-back', 'rodinia-backprop-0-baseline-forward', 'rodinia-backprop-1-tiling-back',
              'rodinia-hotspot-0-baseline', 'rodinia-hotspot-1-tiling', 'rodinia-hotspot-2-pipeline', 'rodinia-hotspot-3-unroll',
              'rodinia-kmeans-0-baseline', 'rodinia-kmeans-1-tiling', 'rodinia-kmeans-2-pipeline', 'rodinia-kmeans-3-unroll',
              'rodinia-knn-0-baseline', 'rodinia-knn-1-tiling', 'rodinia-knn-2-pipeline', 'rodinia-knn-3-unroll',
              'rodinia-knn-4-doublebuffer', 'rodinia-knn-5-coalescing', 'rodinia_cfd_flux_0_baseline_0',
              'rodinia_cfd_step_factor_0_baseline_0', 'rodinia_cfd_step_factor_1_tiling_0', 'rodinia_cfd_step_factor_2_pipeline_0',
              'rodinia_cfd_step_factor_3_unroll_0', 'rodinia_cfd_step_factor_4_doublebuffer_0', 'rodinia_cfd_step_factor_5_coalescing_0',
              'rodinia_dilate_0_baseline_0', 'rodinia_dilate_1_tiling_0', 'rodinia_dilate_2_pipeline_0', 'rodinia_dilate_3_pipeline_0',
              'rodinia_lavaMD_0_baseline', 'rodinia_lavaMD_1_tiling_0', 'rodinia_lavaMD_1_tiling_1', 'rodinia_lavaMD_2_pipeline_0',
              'rodinia_lc_gicov_0_baseline_0', 'rodinia_lc_mgvf_0_baseline_0', 'rodinia_lud_1_tiling_0', 'rodinia_pathfinder_0_baseline_0',
              'rodinia_pathfinder_4_doublebuffer_0', 'rodinia_streamcluster_0_baseline_0', 'rodinia_streamcluster_1_tiling_0',
              'rodinia_streamcluster_2_pipeline_1', 'rodinia_streamcluster_3_doublebuffer_0', 'rodinia_streamcluster_4_coalescing_0',
              'serrano-kalman-filter', 'spcl_example_00', 'spcl_example_01', 'spcl_example_03', 'spcl_example_05']


parser.add_argument('--force_regen', type=bool, default=False) ## must be set to True for the first time to generate the dataset

parser.add_argument('--min_allowed_latency', type=float, default=0.1) ## if latency is less than this, prune the point (used when synthesis is not valid)
EPSILON = 1e-6
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 1e7
parser.add_argument('--normalizer', default=NORMALIZER)
parser.add_argument('--util_normalizer', default=1)
MAX_NUMBER = 1e10
parser.add_argument('--max_number', default=MAX_NUMBER)

norm = 'log2' # 'speedup-log2' 'const' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
parser.add_argument('--norm_method', default=norm)
parser.add_argument('--new_speedup', default=True) # new_speedup: same reference point across all,
                                                    # old_speedup: base is the longest latency and different per kernel

parser.add_argument('--invalid', type = bool, default=False ) # False: do not include invalid designs

parser.add_argument('--encode_log', type = bool, default=False)
v_db = 'v21' # 'v20': v20 database, 'v18': v18 database
parser.add_argument('--v_db', default=v_db) # if set to true uses the db of the new version of the tool: 2020.2

test_kernels = None
parser.add_argument('--test_kernels', default=test_kernels)
target_kernel = None
# target_kernel = 'gemm-blocked'
parser.add_argument('--target_kernel', default=target_kernel)
if target_kernel == None:
    all_kernels = True
else:
    all_kernels = False
parser.add_argument('--all_kernels', type = bool, default=all_kernels)

dataset = 'harp' # machsuite and poly
parser.add_argument('--dataset', default=dataset)

benchmark = ['machsuite', 'poly']
parser.add_argument('--benchmarks', default=benchmark)

tag = 'whole-machsuite-poly'
parser.add_argument('--tag', default=tag)


###################### graph type ######################
graph_type = 'original' # original DAC22 graph
graph_type = 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--graph_type', default=graph_type)

################## model architecture ##################
pragma_as_MLP, type_parallel, type_merge = True, '2l', '2l'
gnn_layer_after_MLP = 1
pragma_MLP_hidden_channels, merge_MLP_hidden_channels = None, None
if 'hierarchy' not in graph_type: ## separate_PT original graph
    gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, True, False, True, 0.1, 154, 7
    model_ver = 'original-PT'
else:
    if pragma_as_MLP:
        if gnn_layer_after_MLP == 1: model_ver = 'pragma_as_MLP'

        if type_parallel == '2l': pragma_MLP_hidden_channels = '[in_D // 2]'
        elif type_parallel == '3l': pragma_MLP_hidden_channels = '[in_D // 2, in_D // 4]'

        if type_merge == '2l': merge_MLP_hidden_channels = '[in_D // 2]'
        elif type_merge == '3l': merge_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
        else: raise NotImplementedError()
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = False, True, True, False, 0.2, 361, 1807  # dropout = 0.1
    else:
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, False, False, True, 0.1, 156, 335
        model_ver = 'hierarchy-PT'

################# one-hot encoder ##################
encoder_path = None
pragma_dim_path = None
encode_edge_position = True
use_encoder = False
if use_encoder:
    encoder_path_list = [f for f in iglob(join(get_root_path(), 'save', 'harp', 'all_kernels', '**'), recursive=True) if f.endswith('.klepto') and 'encoders' in f]
    pragma_dim_path_list = [f for f in iglob(join(get_root_path(), 'save', 'harp', 'all_kernels', '**'), recursive=True) if f.endswith('.klepto') and 'pragma_dim' in f]

    assert len(encoder_path_list) == 1, print(encoder_path_list)
    encoder_path = encoder_path_list[0]
    assert len(pragma_dim_path_list) == 1, print(pragma_dim_path_list)
    pragma_dim_path = pragma_dim_path_list[0]

parser.add_argument('--encoder_path', default=encoder_path)
parser.add_argument('--pragma_dim_path', default=pragma_dim_path)


################ model architecture #################
## edge attributes
parser.add_argument('--encode_edge', type=bool, default=True)
parser.add_argument('--encode_edge_position', type=bool, default=encode_edge_position)

num_layers = 4  # 6
parser.add_argument('--num_layers', type=int, default=num_layers)
parser.add_argument('--num_features', default=num_features)
parser.add_argument('--edge_dim', default=edge_dim)

multi_target = ['perf', 'area'] #, 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
if SUBTASK == 'class':
    multi_target = ['perf']
parser.add_argument('--target', default=multi_target)
parser.add_argument('--MLP_common_lyr', default=0)
gnn_type = 'transformer'
parser.add_argument('--gnn_type', type=str, default=gnn_type)
parser.add_argument('--dropout', type=float, default=dropout)

jkn_mode = 'max'
parser.add_argument('--jkn_mode', type=str, default=jkn_mode)
parser.add_argument('--jkn_enable', type=bool, default=True)
node_attention = True
parser.add_argument('--node_attention', type=bool, default=node_attention)
if node_attention:
    parser.add_argument('--node_attention_MLP', type=bool, default=False)

    separate_P = True
    parser.add_argument('--separate_P', type=bool, default=separate_P)
    separate_icmp = False
    parser.add_argument('--separate_icmp', type=bool, default=separate_icmp)
    separate_T = False
    parser.add_argument('--separate_T', type=bool, default=separate_T)
    separate_pseudo = True
    parser.add_argument('--separate_pseudo', type=bool, default=separate_pseudo)

    if separate_P:
        parser.add_argument('--P_use_all_nodes', type=bool, default=P_use_all_nodes)

## graph auto encoder
parser.add_argument('--gae_T', default = gae_T)
gae_P = False
parser.add_argument('--gae_P', default = gae_P)
if gae_P:
    parser.add_argument('--input_encode', default = False)
    d_type = 'type1'
    parser.add_argument('--decoder_type', default = d_type)

if pragma_as_MLP:
    assert graph_type == 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--gnn_layer_after_MLP', default=gnn_layer_after_MLP) ## number of message passing layers after MLP (pragma as MLP)
parser.add_argument('--pragma_as_MLP', default=pragma_as_MLP)
pragma_as_MLP_list = ['pipeline', 'unroll', 'array_partition']
parser.add_argument('--pragma_as_MLP_list', default=pragma_as_MLP_list)
pragma_scope = 'block'
parser.add_argument('--pragma_scope', default=pragma_scope)
keep_pragma_attribute = False if pragma_as_MLP else True
parser.add_argument('--keep_pragma_attribute', default=keep_pragma_attribute)
pragma_order = 'parallel_and_merge'
parser.add_argument('--pragma_order', default=pragma_order)
pragma_MLP_hidden_channels = '[in_D // 2]'
parser.add_argument('--pragma_MLP_hidden_channels', default=pragma_MLP_hidden_channels)
merge_MLP_hidden_channels = '[in_D // 2]'
parser.add_argument('--merge_MLP_hidden_channels', default=merge_MLP_hidden_channels)


model_path = None
model_path_list = ['/home/ubuntu/val_model_state_dict.pth']
use_pretrain = False
if use_pretrain:
    #base_path = 'models'
    #keyword =  v_db
    #includes = [keyword, model_ver, 'regression']
    #excludes = ['class']
    #model_base_path = '/home/elvouvali/logs/dse_results_v21_2025-12-16T07-58-00.354322/run1/*'
    #model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and 'val' in f]
    #print(model)
    model_path = model_path_list

parser.add_argument('--model_path', default=model_path) ## list of models when used in DSE, if more than 1, ensemble inference must be on

ensemble = 0
ensemble_weights = None
parser.add_argument('--ensemble', type=int, default=ensemble)
parser.add_argument('--ensemble_weights', default=ensemble_weights)
class_model_path = None
if SUBTASK == 'dse':
    keyword =  v_db
    includes = [keyword, model_ver, 'class']
    model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and all(k in f for k in includes)]
    assert len(model) == 1
    class_model_path = model[0]
parser.add_argument('--class_model_path', default=class_model_path)


################ transfer learning #################
feature_extract = False
parser.add_argument('--feature_extract', default=feature_extract) # if set to true GNN encoder (or part of it) will be fixed and only MLP will be trained
if feature_extract:
    parser.add_argument('--random_MLP', default=False) # true: initialize MLP randomly
fix_gnn_layer = 1 ## if none, all layers will be fixed
# fix_gnn_layer = 1 ## number of gnn layers to freeze, feature_extract should be set to True
parser.add_argument('--fix_gnn_layer', default=fix_gnn_layer) # if not set to none, feature_extract should be True
FT_extra = False
parser.add_argument('--FT_extra', default=FT_extra) ## fine-tune only on the new data points


################ training details #################
parser.add_argument('--save_model', type = bool, default=True)
resample = False
val_ratio = 0.05    # 0.15
parser.add_argument('--resample', default=resample) ## when resample is turned on, it will divide the dataset in round-robin and train multiple times to have all the points in train/test set
parser.add_argument('--val_ratio', type=float, default=val_ratio) # ratio of database for validation set
parser.add_argument('--activation', default='elu')
parser.add_argument('--D', type=int, default=64)    
scheduler, warmup, weight_decay = 'cosine', 'linear', 1e-4
parser.add_argument('--weight_decay', type=float, default=weight_decay) ## default=0.0001, larger than 1e-4 didn't help original graph P+T
parser.add_argument("--scheduler", default=scheduler)
parser.add_argument("--warmup", default=warmup)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--random_seed', type=int, default=123)
batch_size = 64
parser.add_argument('--batch_size', type=int, default=batch_size)

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--eval_num_workers', type=int, default=0)
parser.add_argument('--prefetch_factor', type=int, default=1)
parser.add_argument('--persistent_workers', action='store_true')

loss = 'MSE' # RMSE, MSE,
parser.add_argument('--loss', type=str, default=loss)

if model_path == None:
    if TASK == 'regression':
        epoch_num = 20
    else:
        epoch_num = 200
else:
    epoch_num = 400

parser.add_argument('--epoch_num', type=int, default=epoch_num)
parser.add_argument('--sanity_print_n', type=int, default=0)

gpu = 0
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)


################ tiny overfit debug ################
parser.add_argument('--tiny_overfit', action='store_true')
parser.add_argument('--tiny_overfit_kernel', type=str, default='machsuite-gemm-blocked')
parser.add_argument('--tiny_overfit_num_samples', type=int, default=64)
parser.add_argument('--tiny_overfit_batch_size', type=int, default=16)
parser.add_argument('--tiny_overfit_epochs', type=int, default=300)
parser.add_argument('--tiny_overfit_workers', type=int, default=0)
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--load_pretrained', action='store_true')


################# DSE details ##################
explorer = 'exhaustive'
parser.add_argument('--explorer', default=explorer)

model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)

parser.add_argument('--prune_util', default=True) # only DSP and BRAM
parser.add_argument('--prune_class', default=True)

parser.add_argument('--print_every_iter', type=int, default=100)

plot = True
parser.add_argument('--plot_pred_points', type=bool, default=plot)

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

# FLAGS = parser.parse_args([])

FLAGS = parser.parse_args()

if FLAGS.tiny_overfit:
    FLAGS.force_regen = False
    FLAGS.target_kernel = FLAGS.tiny_overfit_kernel
    FLAGS.all_kernels = False

    FLAGS.batch_size = FLAGS.tiny_overfit_batch_size
    FLAGS.epoch_num = FLAGS.tiny_overfit_epochs

    FLAGS.val_ratio = 0.0
    FLAGS.dropout = 0.0
    FLAGS.weight_decay = 0.0
    FLAGS.scheduler = None
    FLAGS.warmup = None

    # Important:
    # keep model_path during inference sanity-check,
    # but disable pretrained loading during training tiny-overfit runs
    if FLAGS.subtask == 'train':
        FLAGS.model_path = None

    FLAGS.save_model = True
    FLAGS.model_tag = f"tiny_overfit_{FLAGS.tiny_overfit_kernel}"

