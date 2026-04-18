import csv
import pandas as pd
import numpy as np
import re
import os
import glob
import subprocess
import tempfile
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch



# def preprocess_csv(csv_file):
#     df = pd.read_csv(csv_file)

#     # Keep only the desired device (MPSoC UltraScale+ ZCU104) and (100 MHz)
#     df_device = df[(df['Device'] == 'xczu7ev-ffvc1156-2-e') & (df['Clock_Period_nsec'] == 10.00)]
#     df_device.drop(columns=['Device', 'Clock_Period_nsec'], inplace=True)

#     # Delete all the columns that are empty
#     nonempty_cols = [col for col in df_device.columns if all(df_device[col] != 'NDIR')]
#     df_filter = df_device[nonempty_cols]

#     utilization_cols = ['BRAM_Utilization_percentage', 'DSP_Utilization_percentage',
#                         'FF_Utilization_percentage', 'LUT_Utilization_percentage']

#     # Set Latency to 0 where utilization percentage is greater than 100 
#     # and where latency is unrealistic (e.g. 10^6 msec)
#     overutil_mask = (df_filter[utilization_cols] >= 100).any(axis=1)
#     too_slow_mask = df_filter['Latency_msec'] > 100000
#     df_filter.loc[overutil_mask | too_slow_mask, 'Latency_msec'] = 0

#     # Convert 0% to 1% in utilization percentages
#     df_filter[utilization_cols] = df_filter[utilization_cols].replace(0, 1)

#     # Average Resource Usage (ARU) = (FF_% + BRAM_% + DSP_% + LUT_%) / 4
#     df_filter['Area'] = df_filter[utilization_cols].sum(axis=1) / 4.0

#     # Valid designs: Latency_msec > 0
#     valid_mask = df_filter['Latency_msec'] > 0
#     df_valid = df_filter[valid_mask].copy()

#     if df_valid.empty:
#         print(f"No valid designs in {csv_file} after filtering.")
#         return
 
#     # Initialize columns
#     df_filter['is_pareto'] = False

#     perf = df_valid['Latency_msec'].to_numpy()
#     area = df_valid['Area'].to_numpy()

#     # Normalize
#     eps = 1e-8
#     perf_min, perf_max = perf.min(), perf.max()
#     area_min, area_max = area.min(), area.max()
#     perf_n = (perf - perf_min) / (perf_max - perf_min + eps)
#     area_n = (area - area_min) / (area_max - area_min + eps)

#     # Pareto frontier on normalized space
#     pareto_mask = pareto_front_2d(perf_n, area_n)

#     # Distance to pareto front (performance, area)
#     perf_p = perf_n[pareto_mask]
#     area_p = area_n[pareto_mask]
#     d = np.empty_like(perf_n)
#     for i in range(len(perf_n)):
#         d[i] = np.min(np.sqrt((perf_n[i] - perf_p)**2 + (area_n[i] - area_p)**2))

#         # ADRS --> pareto front

#     # Normalize distance and map to weights
#     d_max = d.max()
#     d_norm = d / (d_max + eps)          

#     gamma = 2.0
#     w_min_valid = 0.1
#     w_valid = w_min_valid + (1.0 - w_min_valid) * (1.0 - d_norm)**gamma 
#     # close to pareto ~ 1, valid but far from pareto ~ 0.1, invalid = 0.01

#     # Attach weights and Pareto flags
#     df_valid['Weight'] = w_valid
#     is_pareto = np.zeros(len(df_valid), dtype=bool)
#     is_pareto[pareto_mask] = True
#     df_valid['is_pareto'] = is_pareto

#     filename = os.path.basename(csv_file)
#     out_path = os.path.join(
#         os.path.expanduser('~/Desktop/Thesis/Data4LLMPrompting/preprocessed_CSVS'),
#         f'preprocessed-{filename}'
#     )
#     df_valid.to_csv(out_path, index=False)




# 2D pareto plot ---> X = Latency_msec , Y = Total Resource Utilization (BRAM, FF, LUT, DSP)
#                     min is better in both

def pareto_front_2d(x, y):
    """
    Return a boolean mask indicating which points are on the Pareto frontier assuming we want to MINIMIZE both x and y.

    A point i is dominated if there exists some point j such that:
        x_j <= x_i and y_j <= y_i, and at least one is strictly <
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i?
            if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                is_pareto[i] = False
                break

    return is_pareto





# Weight distribution helper 
def print_weight_stats(kernel_csv_name, w):
    w = np.asarray(w)
    print(f"\n[Weight stats] {kernel_csv_name}")
    print(f"  n = {len(w)}")
    print(f"  min / max : {w.min():.3f} / {w.max():.3f}")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  p{int(q*100):2d}: {np.quantile(w, q):.3f}")


def preprocess_csv(csv_file,
                   out_dir=os.path.expanduser('/home/elvouvali/Data4LLMPrompting/preprocessed_CSVS'),
                   w_min_valid=0.1,
                   gamma=2.0):
    df = pd.read_csv(csv_file)

    # Filter device and clock
    df_device = df[(df['Device'] == 'xczu7ev-ffvc1156-2-e') &
                   (df['Clock_Period_nsec'] == 10.00)].copy()
    df_device.drop(columns=['Device', 'Clock_Period_nsec'], inplace=True)

    # Drop empty columns
    nonempty_cols = [col for col in df_device.columns
                     if all(df_device[col] != 'NDIR')]
    df_filter = df_device[nonempty_cols]

    utilization_cols = [
        'BRAM_Utilization_percentage',
        'DSP_Utilization_percentage',
        'FF_Utilization_percentage',
        'LUT_Utilization_percentage'
    ]

    # Invalidate over-utilized or insane-latency designs
    overutil_mask = (df_filter[utilization_cols] >= 100).any(axis=1)
    too_slow_mask = df_filter['Latency_msec'] > 100000
    df_filter.loc[overutil_mask | too_slow_mask, 'Latency_msec'] = 0

    # Convert 0% -> 1% in utilization
    df_filter[utilization_cols] = df_filter[utilization_cols].replace(0, 1)

    # Define Area = mean of utilization %
    df_filter['Area'] = df_filter[utilization_cols].sum(axis=1) / 4.0

    # Keep only valid designs
    valid_mask = df_filter['Latency_msec'] > 0
    df_valid = df_filter[valid_mask].copy()

    if df_valid.empty:
        print(f"[preprocess_csv] No valid designs in {csv_file}")
        return

    perf = df_valid['Latency_msec'].to_numpy()
    area = df_valid['Area'].to_numpy()

    # Normalize (0..1) for distance computations
    eps = 1e-8
    perf_min, perf_max = perf.min(), perf.max()
    area_min, area_max = area.min(), area.max()
    perf_n = (perf - perf_min) / (perf_max - perf_min + eps)
    area_n = (area - area_min) / (area_max - area_min + eps)

    # Pareto front on normalized space
    pareto_mask = pareto_front_2d(perf_n, area_n)

    x_front = perf_n[pareto_mask]
    y_front = area_n[pareto_mask]

    # Sort frontier by performance (x) to get a polyline
    order = np.argsort(x_front)
    x_front = x_front[order]
    y_front = y_front[order]

    # best frontier point = most left + bottom (min area + latency)
    # We have normalized to [0,1] so we can just pick the min lexicographically
    best_idx = np.argmin(x_front + y_front)   
    x_best = x_front[best_idx]
    y_best = y_front[best_idx]

    # For each design, compute:
    #     - d_perp: perpendicular distance to closest frontier segment
    #     - d_best: distance to the best frontier point
    n = len(perf_n)
    d_perp = np.zeros(n)
    d_best = np.zeros(n)

    for i in range(n):
        px, py = perf_n[i], area_n[i]

        # Euclidean distance to best frontier point
        d_best[i] = np.sqrt((px - x_best)**2 + (py - y_best)**2)
    
        # if pareto front has only one point compute the distance from it
        if len(x_front) == 1 :
            min_seg_dist = np.min(np.sqrt((px - x_front)**2 + (py - y_front)**2))

        else :
            # perpendicular distance to frontier 
            min_seg_dist = np.inf   
            # approximate the continuous frontier with line segments between consecutive pareto points
            for k in range(len(x_front) - 1):
                x1, y1 = x_front[k], y_front[k]
                x2, y2 = x_front[k + 1], y_front[k + 1]

                vx, vy = x2 - x1, y2 - y1   # vector along the segment
                wx, wy = px - x1, py - y1   # vector from segment start to the design point
                seg_len2 = vx*vx + vy*vy + eps  

                # dot product v . w = vx*wx + vy*wy
                t = (vx*wx + vy*wy) / seg_len2  # projection scalar from design point to segment
                t = np.clip(t, 0.0, 1.0)    # constrain the projection to stay within the segment

                proj_x = x1 + t * vx
                proj_y = y1 + t * vy

                dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
                if dist < min_seg_dist:
                    min_seg_dist = dist

        d_perp[i] = min_seg_dist

    # Combine distances:
    #     - d_perp: how far from the front 
    #     - d_best: how far from the best region of the front
    alpha = 1.0
    beta = 1.0
    d_total = alpha * d_perp + beta * d_best

    # Convert d_total to weights in [w_min_valid, 1] (smaller distance --> larger weight)
    d_min = d_total.min()
    d_max = d_total.max() + eps

    d_norm = (d_total - d_min) / d_max  # in [0,1]
    w_valid = w_min_valid + (1.0 - w_min_valid) * (1.0 - d_norm)**gamma # gamma > 1 --> emphasizes points close to good pareto

    # Pareto points get weight exactly 1.0
    w_valid[pareto_mask] = 1.0

    # Attach weights and Pareto flags
    df_valid['Weight'] = w_valid
    is_pareto = np.zeros(len(df_valid), dtype=bool)
    is_pareto[pareto_mask] = True
    df_valid['is_pareto'] = is_pareto

    filename = os.path.basename(csv_file)
    out_path = os.path.join(out_dir, f'preprocessed-{filename}')
    os.makedirs(out_dir, exist_ok=True)
    df_valid.to_csv(out_path, index=False)
    print(f"[preprocess_csv] Saved preprocessed CSV to: {out_path}")

    # Weight distribution for sanity check
    w = w_valid
    print(f"[Weight stats] {os.path.basename(csv_file)}")
    print(f"  n = {len(w)}")
    print(f"  min / max : {w.min():.3f} / {w.max():.3f}")
    for p in [10, 25, 50, 75, 90]:
        q = np.percentile(w, p)
        print(f"  p{p:02d}: {q:.3f}")




def plot_pareto_for_kernel(
    csv_file,
    output_dir=os.path.expanduser('/home/elvouvali/Data4LLMPrompting/pareto_per_kernel')
):
    df = pd.read_csv(csv_file)

    # Keep only valid design points (Latency_msec > 0) 
    # Utilization percentages >= 100% have given Latency_msec = 0 in preprocess_csv
    df_valid = df[df['Latency_msec'] > 0].copy()

    if df_valid.empty:
        print(f"No valid points in {csv_file} after filtering.")
        return

    perf = df_valid['Latency_msec'].to_numpy()
    # We have defined 'Area' column as (FF_% + BRAM_% + DSP_% + LUT_%) / 4.0 in preprocess_csv
    area = df_valid['Area'].to_numpy()
    weights = df_valid['Weight'].to_numpy()

    # Pareto frontier on normalized space (minimize latency and resource util)
    pareto_mask = pareto_front_2d(perf, area)
    perf_p = perf[pareto_mask]
    area_p = area[pareto_mask]

    # Sort frontier points by latency so the line looks nice
    order = np.argsort(perf_p)
    perf_p = perf_p[order]
    area_p = area_p[order]

    # Create plot
    plt.figure(figsize=(8, 6))

    # All valid designs, colored by weight
    # higher weight --> closer to Pareto 
    sc = plt.scatter(
        perf,
        area,
        c=weights,
        cmap='viridis',      
        s=20,
        alpha=0.7,
        label='Design points (colored by Weight)',
    )

    plt.scatter(
        perf_p,
        area_p,
        color='red',
        edgecolors='black',
        s=40,
        label='Pareto frontier points',
        zorder=3,
    )

    plt.plot(perf_p, area_p, color='red', linewidth=2, alpha=0.8, zorder=2)
    # Colorbar for weights
    cbar = plt.colorbar(sc)
    cbar.set_label("Weight (higher = closer to Pareto)", rotation=90)

    plt.xlabel('Performance')
    plt.ylabel('Area\n(BRAM + DSP + FF + LUT) / 4')
    title = os.path.basename(csv_file)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    # Build output filename: pareto-<kernel>.png
    base = os.path.splitext(os.path.basename(csv_file))[0]
    if base.startswith('preprocessed-'):
        base = base[len('preprocessed-'):]
    out_path = os.path.join(output_dir, f'pareto-{base}.png')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved Pareto plot to: {out_path}")


def plot_pareto_for_kernel_LLM_pred(
    csv_file,
    output_dir=os.path.expanduser('/home/elvouvali/Data4LLMPrompting/pareto_per_kernel'),
    llm_latency_msec=None,
    llm_bram_pct=None,
    llm_dsp_pct=None,
    llm_ff_pct=None,
    llm_lut_pct=None,
):
    """
    1. Plots all design points colored by their Weight (proximity to Pareto).
    2. Draws the Pareto Frontier line.
    3. Overlays the LLM prediction point if provided.
    """

    df = pd.read_csv(csv_file)

    # Keep only valid design points (Latency_msec > 0)
    df_valid = df[df['Latency_msec'] > 0].copy()

    if df_valid.empty:
        print(f"No valid points in {csv_file} after filtering.")
        return

    perf = df_valid['Latency_msec'].to_numpy()
    area = df_valid['Area'].to_numpy()
    weights = df_valid['Weight'].to_numpy()

    # Calculate Pareto frontier
    pareto_mask = pareto_front_2d(perf, area)
    perf_p = perf[pareto_mask]
    area_p = area[pareto_mask]

    # Sort for a clean line plot
    order = np.argsort(perf_p)
    perf_p = perf_p[order]
    area_p = area_p[order]

    plt.figure(figsize=(10, 7))

    # 1. Plot all points colored by Weight
    sc = plt.scatter(
        perf,
        area,
        c=weights,
        cmap='viridis',
        s=25,
        alpha=0.6,
        label='Design Space (Weighted)'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Weight (Reward Magnitude)", rotation=90)

    # 2. Plot the Pareto Frontier
    plt.plot(perf_p, area_p, color='red', linewidth=2.5, label='Pareto Frontier', zorder=5)
    plt.scatter(perf_p, area_p, color='red', s=50, edgecolors='black', zorder=6)

    # 3. Overlay LLM Prediction
    llm_metrics = [llm_latency_msec, llm_bram_pct, llm_dsp_pct, llm_ff_pct, llm_lut_pct]
    if all(m is not None for m in llm_metrics):
        llm_area = (llm_bram_pct + llm_dsp_pct + llm_ff_pct + llm_lut_pct) / 4.0
        plt.scatter(
            [llm_latency_msec],
            [llm_area],
            marker='X',
            s=150,
            color='white',
            edgecolors='black',
            linewidth=1.5,
            label='LLM Prediction',
            zorder=10
        )
        
    plt.xlabel('Performance (Latency msec)')
    plt.ylabel('Normalized Area\n(BRAM + DSP + FF + LUT) / 4')

    title = f"Pareto Analysis: {os.path.basename(csv_file)}"
    plt.title(title, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right')

    # Filename handling
    base = os.path.splitext(os.path.basename(csv_file))[0]
    if base.startswith('preprocessed-'):
        base = base[len('preprocessed-'):]
    out_path = os.path.join(output_dir, f'pareto-{base}.png')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Pareto plot with LLM prediction saved to: {out_path}")


def main():
    in_dir = "/home/ubuntu/Data4LLMPrompting/CSVS"
    out_dir = "/home/ubuntu/Data4LLMPrompting/preprocessed_CSVS"

    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    print(f"Found {len(csv_files)} CSV files in {in_dir}")

    ok = 0
    failed = 0

    for i, csv_file in enumerate(csv_files, start=1):
        name = os.path.basename(csv_file)
        print(f"\n[{i}/{len(csv_files)}] Processing {name}")
        try:
            preprocess_csv(csv_file, out_dir=out_dir)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")

    print("\nDone.")
    print(f"Processed successfully: {ok}")
    print(f"Failed: {failed}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()