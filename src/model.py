#-----------------------------------------------------------
#                       model.py
#-----------------------------------------------------------

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GlobalAttention, JumpingKnowledge, TransformerConv, GCNConv
from torch_geometric.nn import global_add_pool
from torch.nn import Sequential, Linear, ReLU

from config import FLAGS
from saver import saver
from utils import MLP, _get_y_with_target, MLP_multi_objective
from nn_att import MyGlobalAttention
from collections import OrderedDict, defaultdict
from typing import Dict, Any, List, Tuple


class Net(nn.Module):
    def __init__(self, in_channels, edge_dim = 0, init_pragma_dict = None, task = FLAGS.task, num_layers = FLAGS.num_layers, D = FLAGS.D, target = FLAGS.target): # in_channels: node feature dimension (num_features=153) , edge_dim: edge feature dimension (335) , D : hidden width (64)
          super(Net, self).__init__()

          self.MLP_version = 'multi_obj'  if len(FLAGS.target) > 1 else  'single_obj' # single-head MLP
          # gnn_type determines the graph message passing operator
          if FLAGS.gnn_type == 'gat':
              conv_class = GATConv
          elif FLAGS.gnn_type == 'gcn':
              conv_class = GCNConv
          elif FLAGS.gnn_type == 'transformer':
              conv_class = TransformerConv  # graph transformer layer --> x' = x + Σ*a*W*x , where a : multi-head dot product attention coefficients
                                            # it supports edge_dim so edge_features can influence attention/messages --> x' = x + Σ*a*(W*x + W*e)
          else:
              raise NotImplementedError()

          if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
              self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim, dropout=FLAGS.dropout)  # builds the first graph conv layer 153 --> 64(=D)
              # dropout (propability) --> stochastically drops activations/attention
              # ingests x, edge_index, edge_attr
          else:
              self.conv_first = conv_class(in_channels, D)


          self.num_conv_layers = num_layers - 1 # 5
          num_layers += FLAGS.gnn_layer_after_MLP # 1 layer after MLP : size 64->64
          self.conv_layers = nn.ModuleList()

          for _ in range(num_layers - 1):
              if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                  conv = conv_class(D, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
              else:
                  conv = conv_class(D, D)
              self.conv_layers.append(conv) # list to hold the 5 conv layers (pre-MLP, pre-pragma, after the first conv) --> all hidden->hidden , size 64->64


          if FLAGS.gae_T: # graph auto encoder for 'T' (targets/pragma vector)
              if FLAGS.separate_T:
                  self.gae_transform_T = nn.ModuleDict()
                  for gname, feat_dim in init_pragma_dict.items():
                      self.gae_transform_T['all'] = Linear(feat_dim[1], D // 8) # maps the per-graph pragma vector into a compact code of size D//8
                  channels = [D // 2, D // 4]
                  self.decoder_T = MLP(D, D // 8,
                              activation_type=FLAGS.activation,
                              hidden_channels=channels,
                              num_hidden_lyr=len(channels)) # maps the graph embedding (size D) to the same D/8 space
                                                            # builds a decoder that tries to reconstruct the pragma vector from the graph embedding (via cosine loss)
                                                            # self-supervision to keep embeddings informative about pragmas.

          if FLAGS.gae_P: # graph auto encoder for 'P' (inputs / context nodes)
              out_channels = in_channels
              if FLAGS.input_encode:
                  self.gate_input = Linear(in_channels, 2 * D)  # projects raw node features x, then global-pools them to a graph code.
                  out_channels = 2 * D

              if FLAGS.decoder_type == 'type1':
                  decoder_arch = []
              elif FLAGS.decoder_type == 'type2':
                  decoder_arch = [D, 2 * D, out_channels]
              self.decoder_P = MLP(D, out_channels, activation_type = FLAGS.activation,
                              hidden_channels = decoder_arch,
                              num_hidden_lyr = len(decoder_arch)) # tries to reconstruct that pooled input code from the graph embedding of size D (again via cosine loss) => the graph embedding keeps information about the input program features
              if FLAGS.decoder_type == 'None':
                  for name, param in self.decoder_P.named_parameters():
                      print(name)
                      param.requires_grad = False

          if FLAGS.gae_T or FLAGS.gae_P:
              self.gae_sim_function = nn.CosineSimilarity()
              self.gae_loss_function = nn.CosineEmbeddingLoss() # this loss tries to maximize the cosine similarity between the encoder output (code in D/8 from pragma vector) and its decoder output (code in D/8 from graph embedding)
                                                                # the graph embedding keeps pragma semantics because it must reconstruct the pragma code => “self-supervision” : internal signals (the pragma settings) as supervision to structure the embedding, not an external label.

          # Jumping Knowledge aggregates representations from multiple layers --> max_pooling (max{x1,..,xn}) across layer outputs => stabilizes the deep GNN and avoids over-smoothing
          self.jkn = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=FLAGS.num_layers)  # or num_layers = 2

          self.task = task

          if task == 'regression':  # predicting performance => we want a single scalar output per graph
              self.out_dim = 1
              self.MLP_out_dim = 1
              self.loss_function = nn.MSELoss()
          else:
              self.out_dim = 2
              self.MLP_out_dim = 2
              self.loss_function = nn.CrossEntropyLoss()

          if FLAGS.node_attention:
          # The separate_* create multiple independent attention heads --> each one pools node embeddings into a graph embedding with soft attention weights learned by a small gate network
              if FLAGS.separate_T:
                  self.gate_nn_T = self.node_att_gate_nn(D) # builds a tiny MLP, a gate network that maps each D-dim node embedding to 1 logit (score)
                  self.glob_T = MyGlobalAttention(self.gate_nn_T, None) # implements the attention pooling
              if FLAGS.separate_P:
                  self.gate_nn_P = self.node_att_gate_nn(D)
                  self.glob_P = MyGlobalAttention(self.gate_nn_P, None)
              if FLAGS.separate_pseudo: ## for now, only pseudo node for block
                  self.gate_nn_pseudo_B = self.node_att_gate_nn(D)
                  self.glob_pseudo_B = MyGlobalAttention(self.gate_nn_pseudo_B, None)
              if FLAGS.separate_icmp:
                  self.gate_nn_icmp = self.node_att_gate_nn(D)
                  self.glob_icmp = MyGlobalAttention(self.gate_nn_icmp, None)


          if 'regression' in self.task:
              _target_list = target
              if not isinstance(FLAGS.target, list):
                  _target_list = [target]
              self.target_list = [t for t in _target_list]
          else:
              self.target_list = ['perf']

          if FLAGS.node_attention:
              dim = FLAGS.separate_T + FLAGS.separate_P + FLAGS.separate_pseudo + FLAGS.separate_icmp # concatenate the multiple attention headouts
              in_D = dim * D  # compute the regressor's input width, here : 64*2=128
          else:
              in_D = D
          if D > 64:
              hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
          else:
              hidden_channels = [D // 2, D // 4, D // 8]  # --> hidden sizes : [32,16,8]

          if self.MLP_version == 'single_obj':
              self.MLPs = nn.ModuleDict()
              for target in self.target_list:
                  self.MLPs[target] = MLP(in_D, self.MLP_out_dim, activation_type=FLAGS.activation,
                                          hidden_channels=hidden_channels,
                                          num_hidden_lyr=len(hidden_channels))  # regressor MLP --> MLPs['perf'] : MLP mapping in_D (128) --> MLP_out_dim (1) with hidden [32,16,8] and ELU activations
          else:
              self.MLPs = MLP_multi_objective(in_D, self.MLP_out_dim, activation_type=FLAGS.activation,
                                      hidden_channels=hidden_channels,
                                      objectives=self.target_list,
                                      num_common_lyr=FLAGS.MLP_common_lyr)

          # --- pragma as MLP (only pipeline, unroll and array_partition) ---
          if FLAGS.pragma_as_MLP:
              # Expect exactly these two:
              self.pragma_as_MLP_list = FLAGS.pragma_as_MLP_list  # ['pipeline','unroll', 'array_partition']
              assert set(self.pragma_as_MLP_list) <= {'pipeline', 'unroll', 'array_partition'}, \
                  f"Unexpected pragma types: {self.pragma_as_MLP_list}"
              self.MLPs_per_pragma = nn.ModuleDict()
              for kind in self.pragma_as_MLP_list:
                  if kind in ('pipeline', 'unroll'):
                      extra_dims = 1      # one scalar (II or FACTOR)
                  elif kind == 'array_partition':
                      extra_dims = 3      # [type, factor, dim]
                  else:
                      raise NotImplementedError(f"Unknown pragma kind {kind}")
                  
                  in_D = D + extra_dims
                  hidden_channels, len_hidden_channels = None, 0
                  if FLAGS.pragma_MLP_hidden_channels is not None:
                      hidden_channels = eval(FLAGS.pragma_MLP_hidden_channels)
                      len_hidden_channels = len(hidden_channels)
                  # 2 tiny per pragma node MLPs that transform node embeddings (D-dim) using the pragma values (scalar) at that node’s block scope , D + extra_dims --> D
                  self.MLPs_per_pragma[kind] = MLP(in_D, D, activation_type=FLAGS.activation,
                                                  hidden_channels=hidden_channels, num_hidden_lyr=len_hidden_channels)

              if FLAGS.pragma_order == 'parallel_and_merge':
                  # 'parallel_and_merge' method lets the model learn interactions between II and FACTOR rather than committing to a fixed sequential order.
                  merge_in = D * len(self.pragma_as_MLP_list)  # It runs both per-pragma MLPs in parallel (pipeline and unroll), concatenates their outputs (2*D) at nodes in scope
                  hidden_channels = eval(FLAGS.merge_MLP_hidden_channels)
                  self.MLPs_per_pragma['merge'] = MLP(merge_in, D, activation_type=FLAGS.activation,
                                                      hidden_channels=hidden_channels, num_hidden_lyr=len(hidden_channels)) # passes the concatenation through a merge MLP to get back to D , 2*D --> D



    def node_att_gate_nn(self, D):  # constructs a tiny MLP, the gate network that computes a scalar score per node from a node’s embedding (size D)
          if FLAGS.node_attention_MLP:
              return MLP(D, 1,
                      activation_type=FLAGS.activation_type,
                      hidden_channels=[D // 2, D // 4, D // 8],
                      num_hidden_lyr=3)
          else:
              return Sequential(Linear(D, D), ReLU(), Linear(D, 1)) # two-layer gate : Linear(D->D), ReLU, Linear(D->1) , Output shape : [N, 1]

    # If gae_T = True : encoded_g = Linear(pragmas) and decoded_out = Decoder(graph_embedding) => pushes the graph embedding to retain information predictive of the pragma vector.
    # If gae_P = True : encoded_g = pooled(input features) and decoded_out = Decoder(graph_embedding) => pushes the graph embedding to retain information predictive of input program features.
    def cal_gae_loss(self, encoded_g, decoded_out):
          target = torch.ones(len(encoded_g), device=FLAGS.device)  ## for similarity, use the negative form for dissimilarity
          target.requires_grad = False
          gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
          return gae_loss

    def mask_emb(self, out, non_zero_ids):
          out = out.permute((1, 0))
          out = out * non_zero_ids  # non_zero_ids : [N, 1] --> 1.0 for nodes in scope and 0.0 otherwise
          out = out.permute((1, 0))

          return out  # [N, D]

    def apply_pragma_mlp(self, mlp_pragma, node_emb, scope_nodes, pragma_tensor, kind):
          """
          node_emb: [N, D]
          kind: 'pipeline' or 'unroll' or 'merge'
          pragma_tensor:
            - if kind in {'pipeline','unroll'}: X_pragma_per_node [N,2] ([:,0]=II, [:,1]=FACTOR)
            - if kind == 'merge': concatenated per-pragma outputs [N, D*len(pragmas)]
          scope_nodes: [N,1] (1.0 for nodes in scope, 0.0 otherwise)
          """
          non_scope_nodes = torch.sub(1, scope_nodes)

          if kind == 'merge':
              # mlp over the concatenated per-pragma outputs, only at scoped nodes
              mlp_inp = self.mask_emb(pragma_tensor, non_zero_ids=scope_nodes)
              mlp_out = mlp_pragma(mlp_inp)
              return self.mask_emb(node_emb, non_zero_ids=non_scope_nodes) + \
                    self.mask_emb(mlp_out, non_zero_ids=scope_nodes)

          # Single-scalar option appended to node emb
          if kind == 'pipeline':
              option = pragma_tensor[:, 0:1]    # II
          elif kind == 'unroll':
              option = pragma_tensor[:, 1:2]    # FACTOR
          elif kind == 'array_partition':
              option = pragma_tensor[:, 2:5]    # [type, factor, dim]
          else:
              raise NotImplementedError(f"Unknown pragma kind {kind}")
    
          mlp_inp = torch.cat((node_emb, option), dim=1)  # concatenates the scalar (II / factor) to the node embedding
          mlp_out = mlp_pragma(self.mask_emb(mlp_inp, non_zero_ids=scope_nodes))

          if FLAGS.pragma_order == 'sequential':
              # write back only on scoped nodes, keep others
              return self.mask_emb(node_emb, non_zero_ids=non_scope_nodes) + \
                    self.mask_emb(mlp_out,  non_zero_ids=scope_nodes)
          elif FLAGS.pragma_order == 'parallel_and_merge':
              # caller will concatenate these for a later 'merge'
              return self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
          else:
              raise NotImplementedError()


    '''
    Runs the same way as the forward function up to the pooled graph representation (out_embed)
    Returns : out_embed (2d)
    encoder-only: returns just the pooled graph representation
    '''
    def _graph_embed(self, data):
          # x : [N, in_channels] = [N, num_features] = [N, F] --> node features (one-hot encoded)
          # edge_index : [2, E] = [2, no_of_edges] (the 2 rows hold source and destination node indices for each edge) --> graph structure
          # edge_attr : [E, edge_dim] --> one feature vector per edge
          # batch : [N] --> which graph each node belongs to in a mini-batch (B graphs in the batch)
          x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
          pragmas = getattr(data, 'pragmas', None)
          if hasattr(data, 'kernel'):
              gname = data.kernel[0]
          # X_pragma_per_node [N_nodes, 2]: the local [II, FACTOR] values aligned to pragma-scope pseudo nodes
          # X_pragmascopenids [N_nodes,1]: 1.0 where the pragma should apply, 0.0 elsewhere
          if hasattr(data, 'X_pragma_per_node'):
              X_pragma_per_node = data.X_pragma_per_node
          outs = []
          out_dict = OrderedDict()
          if FLAGS.activation == 'relu':
              activation = F.relu
          elif FLAGS.activation == 'elu':
              activation = F.elu
          else:
              raise NotImplementedError()

          # first conv
          if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
              out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr)) # apply ELU activation on first layer TransformerConv
          else:
              out = activation(self.conv_first(x, edge_index))
          outs.append(out)

          # remaining convs
          for i in range(self.num_conv_layers):
              conv = self.conv_layers[i]
              if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                  out = conv(out, edge_index, edge_attr=edge_attr)
              else:
                  out = conv(out, edge_index)
              if i != len(self.conv_layers) - 1:  # apply activation on all the graph convs but the very last one (on 4 TransformerConv)
                  out = activation(out)

              outs.append(out)

          if FLAGS.jkn_enable:
              out = self.jkn(outs)  # fuses the layer-wise representations (node embeddings) into a single tensor of shape [N, D], jkn_mode=max

          # pragma as MLP
          if FLAGS.pragma_as_MLP:
              assert hasattr(data, 'X_pragma_per_node'), "Missing X_pragma_per_node"
              X_pragma_per_node = data.X_pragma_per_node
              assert X_pragma_per_node.size(-1) == 5, \
                  f"X_pragma_per_node must be [...,5] ([PIPE_II, UNROLL_FACTOR, PARTITION_TYPE, PARTITION_FACTOR, PARTITION_DIM]), got {tuple(X_pragma_per_node.size())}"
              in_merge = None
              for kind in self.pragma_as_MLP_list:  # ['pipeline','unroll']
                  # concatenating node embedding [N,D] with the one scalar option (II / FACTOR) --> [N, D+1]
                  # mask to only pragma-scope nodes (X_pragmascopenids) --> tiny MLP --> updated node embedding [N, D]
                  out_MLP = self.apply_pragma_mlp(self.MLPs_per_pragma[kind],
                                                  out, data.X_pragmascopenids,
                                                  X_pragma_per_node, kind)
                  if FLAGS.pragma_order == 'sequential':
                      out = out_MLP
                  elif FLAGS.pragma_order == 'parallel_and_merge':
                      in_merge = out_MLP if in_merge is None else torch.cat((in_merge, out_MLP), dim=1)
                  else:
                      raise NotImplementedError()

              if FLAGS.pragma_order == 'parallel_and_merge':
                  # concatenation of both per-pragma outputs [N, 2D] --> merge MLP --> updated node embedding [N, D]
                  out = self.apply_pragma_mlp(self.MLPs_per_pragma['merge'],
                                              out, data.X_pragmascopenids,
                                              in_merge, 'merge')

              # post-pragma extra conv layers
              # 1 conv layer after the pragma MLPs to let the network diffuse the local pragma effects through the graph
              for i, conv in enumerate(self.conv_layers[self.num_conv_layers:]):
                  if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                      out = conv(out, edge_index, edge_attr=edge_attr)
                  else:
                      out = conv(out, edge_index)
                  layer = i + self.num_conv_layers
                  if layer != len(self.conv_layers) - 1:
                      out = activation(out)

          # get a graph-level vector
          if FLAGS.node_attention:
              out_gnn = out
              out_g = None
              out_P, out_T = None, None
              if FLAGS.separate_P:
                  # glob_P: a learnable softmax attention over all nodes --> [B, D] : graph vector that emphasizes nodes useful for prediction
                  if FLAGS.P_use_all_nodes:
                      out_P, node_att_scores_P = self.glob_P(out_gnn, batch)
                  else:
                      out_P, node_att_scores_P = self.glob_P(out_gnn, batch, set_zeros_ids=data.X_contextnids)

                  out_g = out_P

              if FLAGS.separate_T:
                  out_T, node_att_scores = self.glob_T(out_gnn, batch, set_zeros_ids=data.X_pragmanids)
                  if out_P is not None:
                      out_g = torch.cat((out_P, out_T), dim=1)
                  else:
                      out_g = out_T

              if FLAGS.separate_pseudo:
                  # glob_pseudo_B: attention pooling that can zero-out everything except pseudo block nodes (via the set_zeros_ids mask) --> [B, D]
                  out_pseudo_B, node_att_scores_pseudo = self.glob_pseudo_B(out_gnn, batch, set_zeros_ids=data.X_pseudonids)
                  if out_g is not None:
                      out_g = torch.cat((out_g, out_pseudo_B), dim=1)
                  else:
                      out_g = out_pseudo_B

              if FLAGS.separate_icmp:
                  out_icmp, node_att_scores_icmp = self.glob_icmp(out_gnn, batch, set_zeros_ids=data.X_icmpnids)
                  if out_g is not None:
                      out_g = torch.cat((out_g, out_icmp), dim=1)
                  else:
                      out_g = out_icmp

              if not FLAGS.separate_P and not FLAGS.separate_T and not FLAGS.separate_pseudo:
                  out_g, node_att_scores = self.glob_T(out_gnn, batch)

              out_embed = out_g # concat [B, D] embeddings --> final graph embedding [B, 2D] (B graphs in a batch)

          else:
              out_embed = global_add_pool(out, batch)

          return out_embed


    # @torch.no_grad()
    def forward_embed(self, data):
        """
        Returns the pooled graph embedding (to keep a frozen encoder of Net)
        """
        self.eval() # frozen encoder
        return self._graph_embed(data)



    '''
    Runs the same way as the forward function up to the final node embeddings representation (out_node_embed)
    '''
    def _node_embed(self, data):
          # x : [N, in_channels] = [N, num_features] = [N, F] --> node features (one-hot encoded)
          # edge_index : [2, E] = [2, no_of_edges] (the 2 rows hold source and destination node indices for each edge) --> graph structure
          # edge_attr : [E, edge_dim] --> one feature vector per edge
          # batch : [N] --> which graph each node belongs to in a mini-batch (B graphs in the batch)
          x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
          pragmas = getattr(data, 'pragmas', None)
          if hasattr(data, 'kernel'):
              gname = data.kernel[0]
          # X_pragma_per_node [N_nodes, 2]: the local [II, FACTOR] values aligned to pragma-scope pseudo nodes
          # X_pragmascopenids [N_nodes,1]: 1.0 where the pragma should apply, 0.0 elsewhere
          if hasattr(data, 'X_pragma_per_node'):
              X_pragma_per_node = data.X_pragma_per_node
          outs = []
          out_dict = OrderedDict()
          if FLAGS.activation == 'relu':
              activation = F.relu
          elif FLAGS.activation == 'elu':
              activation = F.elu
          else:
              raise NotImplementedError()

          # first conv
          if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
              out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr)) # apply ELU activation on first layer TransformerConv
          else:
              out = activation(self.conv_first(x, edge_index))
          outs.append(out)

          # remaining convs
          for i in range(self.num_conv_layers):
              conv = self.conv_layers[i]
              if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                  out = conv(out, edge_index, edge_attr=edge_attr)
              else:
                  out = conv(out, edge_index)
              if i != len(self.conv_layers) - 1:  # apply activation on all the graph convs but the very last one (on 4 TransformerConv)
                  out = activation(out)

              outs.append(out)

          if FLAGS.jkn_enable:
              out = self.jkn(outs)  # fuses the layer-wise representations (node embeddings) into a single tensor of shape [N, D], jkn_mode=max         
          ## pragma as MLP
          if FLAGS.pragma_as_MLP:
              assert hasattr(data, 'X_pragma_per_node'), "Missing X_pragma_per_node"
              X_pragma_per_node = data.X_pragma_per_node
              assert X_pragma_per_node.size(-1) == 5, \
                  f"X_pragma_per_node must be [...,5] ([PIPE_II, UNROLL_FACTOR, PARTITION_TYPE, PARTITION_FACTOR, PARTITION_DIM]), got {tuple(X_pragma_per_node.size())}"
              in_merge = None
              for kind in self.pragma_as_MLP_list:  # ['pipeline','unroll']
                  out_MLP = self.apply_pragma_mlp(self.MLPs_per_pragma[kind],
                                                  out, data.X_pragmascopenids,
                                                  X_pragma_per_node, kind)
                  if FLAGS.pragma_order == 'sequential':
                      out = out_MLP
                  elif FLAGS.pragma_order == 'parallel_and_merge':
                      in_merge = out_MLP if in_merge is None else torch.cat((in_merge, out_MLP), dim=1)
                  else:
                      raise NotImplementedError()

              if FLAGS.pragma_order == 'parallel_and_merge':
                 # merge the two streams back to D using 'merge' MLP
                  out = self.apply_pragma_mlp(self.MLPs_per_pragma['merge'],
                                              out, data.X_pragmascopenids,
                                              in_merge, 'merge')

              for i, conv in enumerate(self.conv_layers[self.num_conv_layers:]):
                  if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                      out = conv(out, edge_index, edge_attr=edge_attr)
                  else:
                      out = conv(out, edge_index)
                  layer = i + self.num_conv_layers
                  if layer != len(self.conv_layers) - 1:
                      out = activation(out)

          out_node_embed = out
 

          return out_node_embed



    def forward_node_embed(self, data):
        """
        Returns the final node embeddings
        """
        return self._node_embed(data)



    '''
    end-to-end model: produces predictions and losses (regression/classification), optionally adds auxiliary GAE losses, and is the function used during training
    '''
    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
        pragmas = getattr(data, "pragmas", None)
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        if hasattr(data, 'X_pragma_per_node'):
            X_pragma_per_node = data.X_pragma_per_node
        outs = []
        out_dict = OrderedDict()
        if FLAGS.activation == 'relu':
            activation = F.relu
        elif FLAGS.activation == 'elu':
            activation = F.elu
        else:
            raise NotImplementedError()


        if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
            out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))
        else:
            out = activation(self.conv_first(x, edge_index))

        outs.append(out)

        for i in range(self.num_conv_layers):
            conv = self.conv_layers[i]
            if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                out = conv(out, edge_index, edge_attr=edge_attr)
            else:
                out = conv(out, edge_index)
            if i != len(self.conv_layers) - 1:
                out = activation(out)

            outs.append(out)

        if FLAGS.jkn_enable:
            out = self.jkn(outs)

        ## pragma as MLP
        if FLAGS.pragma_as_MLP:
            assert hasattr(data, 'X_pragma_per_node'), "Missing X_pragma_per_node"
            X_pragma_per_node = data.X_pragma_per_node
            assert X_pragma_per_node.size(-1) == 5, \
                  f"X_pragma_per_node must be [...,5] ([PIPE_II, UNROLL_FACTOR, PARTITION_TYPE, PARTITION_FACTOR, PARTITION_DIM]), got {tuple(X_pragma_per_node.size())}"
            in_merge = None
            for kind in self.pragma_as_MLP_list:  # ['pipeline','unroll']
                out_MLP = self.apply_pragma_mlp(self.MLPs_per_pragma[kind],
                                                out, data.X_pragmascopenids,
                                                X_pragma_per_node, kind)
                if FLAGS.pragma_order == 'sequential':
                    out = out_MLP
                elif FLAGS.pragma_order == 'parallel_and_merge':
                    in_merge = out_MLP if in_merge is None else torch.cat((in_merge, out_MLP), dim=1)
                else:
                    raise NotImplementedError()

            if FLAGS.pragma_order == 'parallel_and_merge':
                # merge the two streams back to D using 'merge' MLP
                out = self.apply_pragma_mlp(self.MLPs_per_pragma['merge'],
                                            out, data.X_pragmascopenids,
                                            in_merge, 'merge')

            for i, conv in enumerate(self.conv_layers[self.num_conv_layers:]):
                if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                    out = conv(out, edge_index, edge_attr=edge_attr)
                else:
                    out = conv(out, edge_index)
                layer = i + self.num_conv_layers
                if layer != len(self.conv_layers) - 1:
                    out = activation(out)

        if FLAGS.node_attention:
            out_gnn = out
            out_g = None
            out_P, out_T = None, None
            if FLAGS.separate_P:
                if FLAGS.P_use_all_nodes:
                    out_P, node_att_scores_P = self.glob_P(out_gnn, batch)
                else:
                    out_P, node_att_scores_P = self.glob_P(out_gnn, batch, set_zeros_ids=data.X_contextnids)

                out_dict['emb_P'] = out_P
                out_g = out_P

            if FLAGS.separate_T:
                out_T, node_att_scores = self.glob_T(out_gnn, batch, set_zeros_ids=data.X_pragmanids)
                out_dict['emb_T'] = out_T
                if out_P is not None:
                    out_g = torch.cat((out_P, out_T), dim=1)
                else:
                    out_g = out_T

            if FLAGS.separate_pseudo:
                out_pseudo_B, node_att_scores_pseudo = self.glob_pseudo_B(out_gnn, batch, set_zeros_ids=data.X_pseudonids)
                out_dict['emb_pseudo_b'] = out_pseudo_B
                if out_g is not None:
                    out_g = torch.cat((out_g, out_pseudo_B), dim=1)
                else:
                    out_g = out_pseudo_B

            if FLAGS.separate_icmp:
                out_icmp, node_att_scores_icmp = self.glob_icmp(out_gnn, batch, set_zeros_ids=data.X_icmpnids)
                out_dict['emb_icmp'] = out_icmp
                if out_g is not None:
                    out_g = torch.cat((out_g, out_icmp), dim=1)
                else:
                    out_g = out_icmp

            if not FLAGS.separate_P and not FLAGS.separate_T and not FLAGS.separate_pseudo:
                out_g, node_att_scores = self.glob_T(out_gnn, batch)
                out_dict['emb_T'] = out
                if FLAGS.subtask == 'visualize':
                    saver.save_dict({'data': data, 'node_att_scores': node_att_scores},
                                    f'node_att.pickle')

            out = out_g
        else:
            out = global_add_pool(out, batch)
            out_dict['emb_T'] = out

        total_loss = 0
        gae_loss = 0
        if FLAGS.gae_T: # graph auto encoder
            assert pragmas is not None, "Missing `pragmas` in Data for GAE-T path"
            assert FLAGS.separate_T
            gname = 'all'
            encoded_g = self.gae_transform_T[gname](pragmas)
            decoded_out = self.decoder_T(out_dict['emb_T'])
            gae_loss = self.cal_gae_loss(encoded_g, decoded_out)
        if FLAGS.gae_P:
            assert FLAGS.separate_P
            encoded_x = x
            if FLAGS.input_encode:
                encoded_x = self.gate_input(x)
            encoded_g = global_add_pool(encoded_x, batch) ## simple addition of node embeddings for gae

            if FLAGS.decoder_type == 'None': ## turn off autograd:
                decoded_out = self.decoder_P(out_dict['emb_P']).detach()
            else:
                decoded_out = self.decoder_P(out_dict['emb_P']).to(FLAGS.device)
            # gae_loss = (self.gae_loss_function(encoded_g, decoded_out)).mean()
            gae_loss += self.cal_gae_loss(encoded_g, decoded_out)
        if FLAGS.gae_P or FLAGS.gae_T:
            total_loss += torch.abs(gae_loss)

        out_embed = out
        loss_dict = {}

        if self.MLP_version == 'multi_obj':
            out_MLPs = self.MLPs(out_embed)
        for target_name in self.target_list:
            if self.MLP_version == 'multi_obj':
                out = out_MLPs[target_name]
            else:
                out = self.MLPs[target_name](out_embed)
            y = _get_y_with_target(data, target_name)
            if self.task == 'regression':
                target = y.view((len(y), self.out_dim))
                # print('target', target.shape)
                if FLAGS.loss == 'RMSE':
                    loss = torch.sqrt(self.loss_function(out, target))
                    # loss = mean_squared_error(target, out, squared=False)
                elif FLAGS.loss == 'MSE':
                    loss = self.loss_function(out, target)
                else:
                    raise NotImplementedError()
                # print('loss', loss.shape)
            else:
                target = y.view((len(y)))
                loss = self.loss_function(out, target)
            out_dict[target_name] = out
            total_loss += loss
            loss_dict[target_name] = loss


        return out_dict, total_loss, loss_dict, gae_loss

