#-----------------------------------------------------------
#        Graph Embedder (cpp code to graph embedding)
#-----------------------------------------------------------

from gexf_to_pt import gexf_to_pt
from pt_to_gnn_emb import extract_single_embedding
from cpp_to_gexf_deterministic import cpp_to_gexf
import subprocess
import os
import torch
import json
from config import FLAGS

def graph_emb_from_cpp(test_code: str, assigns: dict, unique_id: str):

    base_path = "/home/ubuntu/LLM_predictions/"
    cpp_path = os.path.join(base_path, f"test_{unique_id}.cpp")
    json_path = os.path.join(base_path, f"pred_{unique_id}.json") # Unique JSON
    pt_path = os.path.join(base_path, f"pred_{unique_id}.pt")
    emb_path = f"/home/ubuntu/GNN_embeddings/pred_{unique_id}.pt"
    gexf_path = os.path.join(base_path, f"test_{unique_id}.gexf")
    
    with open(cpp_path, 'w') as f:
        f.write(test_code)
    
    with open(json_path, 'w') as f:
        json.dump(assigns, f)

    g = cpp_to_gexf(
        name=f"test_{unique_id}",
        path=base_path,
        src_ext="cpp",
        out_gexf=gexf_path,
        log=False,
    )
    print(f"OK: nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

    gexf_to_pt(
         gexf_path=gexf_path,
         point_json=json_path,
         out_pt=pt_path,
         key_name=f"test_{unique_id}",
    )
    pt = torch.load(pt_path)
    print(f".pt point OK: shape={pt.x.shape}") # Note: .pt files usually have .x for features

    checkpoint_path = "/home/ubuntu/logs/all_kernels_GNN_train/run1/val_model_state_dict.pth"

    emb = extract_single_embedding(
        pt_path,
        checkpoint_path,
        device=FLAGS.device
        )

    torch.save(emb, emb_path)
    print(f"GNN Embedding OK: shape={emb.shape}")

    ll_path = cpp_path.replace(".cpp", ".ll")
    for p in [cpp_path, json_path, pt_path, emb_path]:
        if os.path.exists(p): os.remove(p)

    return emb



gemv_code = """
#define N 32
#define M 32

extern "C" void gemv(const float A[N][M], const float x[M], float y[N]) {

#pragma HLS array_partition variable=A type=cyclic factor=8 dim=2
#pragma HLS array_partition variable=x type=complete factor=0 dim=1

for (int i=0; i<N; i++) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=0
    float acc=0;
    for (int j=0; j<M; j++) {
#pragma HLS pipeline II=0
#pragma HLS unroll factor=8
        acc += A[i][j] * x[j];
        }
        y[i] = acc;
    }
}
"""

test_code = """
/*
Implementation based on algorithm described in:
A. Danalis, G. Marin, C. McCurdy, J. S. Meredith, P. C. Roth, K. Spafford, V. Tipparaju, and J. S. Vetter.
The scalable heterogeneous computing (shoc) benchmark suite.
In Proceedings of the 3rd Workshop on General-Purpose Computation on Graphics Processing Units, 2010
*/

#include "sort.h"

void local_scan(int bucket[BUCKETSIZE])
{
    int radixID, i, bucket_indx;
L1:    local_1 : for (radixID=0; radixID<SCAN_RADIX; radixID++) {
#pragma HLS pipeline II=auto{_PIPE_L1}
#pragma HLS unroll factor=auto{_UNROLL_L1}
L2:        local_2 : for (i=1; i<SCAN_BLOCK; i++){
#pragma HLS pipeline II=auto{_PIPE_L2}
#pragma HLS unroll factor=auto{_UNROLL_L2}
            bucket_indx = radixID*SCAN_BLOCK + i;
            bucket[bucket_indx] += bucket[bucket_indx-1];
        }
    }
}

void sum_scan(int sum[SCAN_RADIX], int bucket[BUCKETSIZE])
{
    int radixID, bucket_indx;
    sum[0] = 0;
L3:    sum_1 : for (radixID=1; radixID<SCAN_RADIX; radixID++) {
#pragma HLS pipeline II=auto{_PIPE_L3}
#pragma HLS unroll factor=auto{_UNROLL_L3}
        bucket_indx = radixID*SCAN_BLOCK - 1;
        sum[radixID] = sum[radixID-1] + bucket[bucket_indx];
    }
}

void last_step_scan(int bucket[BUCKETSIZE], int sum[SCAN_RADIX])
{
    int radixID, i, bucket_indx;
L4:    last_1:for (radixID=0; radixID<SCAN_RADIX; radixID++) {
#pragma HLS pipeline II=auto{_PIPE_L4}
#pragma HLS unroll factor=auto{_UNROLL_L4}
L5:        last_2:for (i=0; i<SCAN_BLOCK; i++) {
#pragma HLS pipeline II=auto{_PIPE_L5}
#pragma HLS unroll factor=auto{_UNROLL_L5}
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] = bucket[bucket_indx] + sum[radixID];
         }
    }
}

void init(int bucket[BUCKETSIZE])
{
    int i;
L6:    init_1 : for (i=0; i<BUCKETSIZE; i++) {
#pragma HLS pipeline II=auto{_PIPE_L6}
#pragma HLS unroll factor=auto{_UNROLL_L6}
        bucket[i] = 0;
    }
}

void hist(int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int blockID, i, bucket_indx, a_indx;
    blockID = 0;
L7:    hist_1 : for (blockID=0; blockID<NUMOFBLOCKS; blockID++) {
#pragma HLS pipeline II=auto{_PIPE_L7}
#pragma HLS unroll factor=auto{_UNROLL_L7}
L8:        hist_2 : for(i=0; i<4; i++) {
#pragma HLS pipeline II=auto{_PIPE_L8}
#pragma HLS unroll factor=auto{_UNROLL_L8}
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            bucket_indx = ((a[a_indx] >> exp) & 0x3)*NUMOFBLOCKS + blockID + 1;
            bucket[bucket_indx]++;
        }
    }
}

void update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int i, blockID, bucket_indx, a_indx;
    blockID = 0;

L9:    update_1 : for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
#pragma HLS pipeline II=auto{_PIPE_L9}
#pragma HLS unroll factor=auto{_UNROLL_L9}
L10:        update_2 : for(i=0; i<4; i++) {
#pragma HLS pipeline II=auto{_PIPE_L10}
#pragma HLS unroll factor=auto{_UNROLL_L10}
            bucket_indx = ((a[blockID * ELEMENTSPERBLOCK + i] >> exp) & 0x3)*NUMOFBLOCKS + blockID;
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            b[bucket[bucket_indx]] = a[a_indx];
            bucket[bucket_indx]++;
        }
    }
}

void ss_sort(int a[SIZE], int b[SIZE], int bucket[BUCKETSIZE], int sum[SCAN_RADIX]){
    int exp=0;
    int valid_buffer=0;
    #define BUFFER_A 0
    #define BUFFER_B 1

L11:    sort_1 : for (exp=0; exp<32; exp+=2) {
#pragma HLS pipeline II=auto{_PIPE_L11}
#pragma HLS unroll factor=auto{_UNROLL_L11}
        init(bucket);
        if (valid_buffer == BUFFER_A) {
            hist(bucket, a, exp);
        } else {
            hist(bucket, b, exp);
        }

        local_scan(bucket);
        sum_scan(sum, bucket);
        last_step_scan(bucket, sum);

        if (valid_buffer==BUFFER_A) {
            update(b, bucket, a, exp);
            valid_buffer = BUFFER_B;
        } else {
            update(a, bucket, b, exp);
            valid_buffer = BUFFER_A;
        }
    }
    // If trip count is even, buffer A will be valid at the end.
}
"""


assigns_str = """

 {
 "_ARRAY_T_L6": "cyclic", "_ARRAY_F_L6": 2, "_ARRAY_D_L6": 1,
 "_ARRAY_T_L7": "cyclic", "_ARRAY_F_L7": 32, "_ARRAY_D_L7": 1,
 "_ARRAY_T_L5": "complete", "_ARRAY_F_L5": 0, "_ARRAY_D_L5": 1,
 "_PIPE_L1": 0, "_UNROLL_L1": 4,
 "_PIPE_L2": 1, "_UNROLL_L2": 0,
 "_PIPE_L4": 0, "_UNROLL_L4": 4,
 "_PIPE_L9": 1, "_UNROLL_L9": 0,
 "_PIPE_L3": 1, "_UNROLL_L3": 0
 }

"""

try:
    # .strip() removes the extra newlines \n\n
    assigns_dict = json.loads(assigns_str.strip())
except json.JSONDecodeError as e:
    print(f"Failed to parse LLM output as JSON: {e}")
    # Fallback to an empty dict or handle error
    assigns_dict = {}

# graph_emb_from_cpp(test_code, assigns_dict, 0)
