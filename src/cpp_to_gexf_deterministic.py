# Run:   PYTHONHASHSEED=0 python cpp_to_gexf_deterministic.py

import os
import shutil
from os.path import join
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Any, Dict, List, Optional, Tuple
import argparse
import networkx as nx
import graph_gen_deterministic as gg
from insert_placeholders import insert_placeholders
from utils import create_dir_if_not_exists, get_root_path, natural_keys


def _require_pythonhashseed() -> None:
    if os.environ.get("PYTHONHASHSEED", "") == "":
        raise RuntimeError(
            "PYTHONHASHSEED must be set for strict determinism across runs.\n"
            "Example:\n"
            "  PYTHONHASHSEED=0 python cpp_to_gexf_deterministic.py\n"
        )


def _get_node_full_text(ndata: Dict[str, Any]) -> Optional[str]:
    """
    Extract full text from either:
      - ProGraML features dict: {"features": {"full_text": ["..."]}}
      - processed: {"full_text": "..."}
      - stringified features: {"features": "{'full_text': ['...']}"}
    """
    import ast

    if "full_text" in ndata and ndata["full_text"] is not None:
        return str(ndata["full_text"])

    if "features" in ndata:
        feat = ndata["features"]
        if isinstance(feat, dict):
            ft = feat.get("full_text")
            if isinstance(ft, list) and ft:
                return str(ft[0])
        if isinstance(feat, str):
            try:
                obj = ast.literal_eval(feat)
                if isinstance(obj, dict):
                    ft = obj.get("full_text")
                    if isinstance(ft, list) and ft:
                        return str(ft[0])
            except Exception:
                pass
    return None


def _stable_node_key(node: Any, data: Dict[str, Any]) -> Tuple:
    """
    Stable sorting key (does NOT use node id as tie-breaker).
    """
    return (
        int(data.get("function", -1)),
        int(data.get("block", -1)),
        int(data.get("type", -1)),
        str(data.get("text", "")),
        str(_get_node_full_text(data) or ""),
    )


def create_pragma_nodes_no_kernel_info(
    g_nx: nx.MultiDiGraph,
    g_nx_nodes: int,
    for_dict_source: Dict[str, Any],
    for_dict_llvm: Dict[str, Any],
    placeholder_src_file: str,
    log: bool = False,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, int, Dict[str, Any]]]]:
    """
    Deterministic clone of create_pragma_nodes, WITHOUT kernel_info.txt.

    Differences vs your original create_pragma_nodes():
      - No tripcount consistency check
      - No eligible-label filtering
      - Array pragmas supported (parsed from placeholder_src_file)

    Returns:
      new_nodes: [(new_node_id, attrs_dict), ...]
      new_edges: [(u, v, attrs_dict), ...]  # keys/ids assigned later deterministically
    """
    import re

    new_nodes: List[Tuple[int, Dict[str, Any]]] = []
    new_edges: List[Tuple[int, int, Dict[str, Any]]] = []
    next_node_id = int(g_nx_nodes)

    def resolve_llvm_key(src_func_name: str) -> Optional[str]:
        # Deterministic match selection (same logic class as your script)
        exact, suffix, substr = [], [], []
        for key in sorted(for_dict_llvm.keys()):
            m = re.search(r'@([^(]+)\s*\(', key)
            if not m:
                continue
            mangled = m.group(1)
            demangled = mangled
            m2 = re.match(r'_Z(\d+)([A-Za-z_]\w*)', mangled)
            if m2:
                try:
                    nlen = int(m2.group(1))
                    cand = m2.group(2)
                    if len(cand) == nlen:
                        demangled = cand
                except ValueError:
                    pass

            if demangled == src_func_name or mangled == src_func_name:
                exact.append(key)
            elif demangled.endswith(src_func_name) or mangled.endswith(src_func_name):
                suffix.append(key)
            elif src_func_name in demangled or src_func_name in mangled:
                substr.append(key)

        if len(exact) == 1:
            return exact[0]
        if len(suffix) == 1:
            return suffix[0]
        if len(substr) == 1:
            return substr[0]
        if len(for_dict_llvm) == 1:
            return sorted(for_dict_llvm.keys())[0]
        return None

    def find_icmp_node(icmp_inst: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        # Deterministic scan order
        for node, ndata in sorted(g_nx.nodes(data=True), key=lambda nd: _stable_node_key(nd[0], nd[1])):
            ft = _get_node_full_text(ndata)
            if ft == icmp_inst:
                return int(node), int(ndata.get("block", -1)), int(ndata.get("function", -1))
        return None, None, None

    # LOOP PRAGMAS
    for f_name in sorted(for_dict_source.keys()):
        f_content = for_dict_source[f_name]
        if not f_content:
            continue

        llvm_key = resolve_llvm_key(f_name)
        if llvm_key is None:
            continue
        llvm_content = for_dict_llvm.get(llvm_key, {})

        for for_loop_id in sorted(f_content.keys()):
            payload = f_content[for_loop_id]

            if isinstance(payload, dict):
                pragmas = list(payload.get("pragmas", []))
                local_id = payload.get("local_id", for_loop_id)
            else:
                # original: [loop_line, pragmas]
                _, pragmas = payload
                pragmas = list(pragmas)
                local_id = for_loop_id

            if local_id not in llvm_content:
                continue

            icmp_inst = llvm_content[local_id][0]
            node_id, block_id, function_id = find_icmp_node(icmp_inst)
            if node_id is None:
                continue

            # Pragmas are processed in deterministic order
            for pragma in sorted(pragmas):
                tokens = pragma.split()
                if len(tokens) < 3:
                    continue
                pragma_kind = tokens[2].upper()
                if pragma_kind not in gg.PRAGMA_POSITION:
                    continue

                p_dict = {
                    "type": 100,
                    "block": block_id,
                    "function": function_id,
                    "features": {"full_text": [pragma]},
                    "text": pragma_kind,
                }
                new_nodes.append((next_node_id, p_dict))

                e_attr = {"flow": 200, "position": gg.PRAGMA_POSITION[pragma_kind]}
                new_edges.append((node_id, next_node_id, e_attr))
                new_edges.append((next_node_id, node_id, e_attr))

                next_node_id += 1

    # ARRAY PRAGMAS
    if os.path.isfile(placeholder_src_file):
        array_pragmas = gg.get_pragmas_arrays(placeholder_src_file, log=log)
        array_pragmas = sorted(array_pragmas, key=lambda x: (x.get("function") or "", x.get("var") or "", x.get("pragma") or ""))

        for ap in array_pragmas:
            varname = ap["var"]
            pragma_line = ap["pragma"]

            matched_nodes: List[Tuple[Any, Dict[str, Any]]] = []
            decl_candidates: List[Tuple[Any, Dict[str, Any]]] = []

            for node, ndata in g_nx.nodes(data=True):
                ft = _get_node_full_text(ndata)
                if not ft:
                    continue
                if varname not in ft:
                    continue

                matched_nodes.append((node, ndata))
                if "alloca" in ft and "[" in ft and "]" in ft:
                    decl_candidates.append((node, ndata))

            if not matched_nodes:
                continue

            matched_nodes.sort(key=lambda nd: _stable_node_key(nd[0], nd[1]))
            decl_candidates.sort(key=lambda nd: _stable_node_key(nd[0], nd[1]))

            node0, data0 = decl_candidates[0] if decl_candidates else matched_nodes[0]
            block_id = int(data0.get("block", -1))
            function_id = int(data0.get("function", -1))

            p_dict = {
                "type": 100,
                "block": block_id,
                "function": function_id,
                "features": {"full_text": [pragma_line]},
                "text": "ARRAY_PARTITION",
            }
            new_nodes.append((next_node_id, p_dict))

            e_attr = {"flow": 200, "position": gg.PRAGMA_POSITION["ARRAY_PARTITION"]}

            # Attach to all uses (deterministic order)
            for node, _ in matched_nodes:
                new_edges.append((int(node), next_node_id, e_attr))
                new_edges.append((next_node_id, int(node), e_attr))

            next_node_id += 1

    return new_nodes, new_edges


def cpp_to_gexf(
    name: str,
    path: str,
    src_ext: str,
    out_gexf: str,
    log: bool = False,
) -> nx.MultiDiGraph:
    """
    Deterministic pipeline: close to your original cpp_to_gexf, but:
      - Uses gg canonicalization + relabeling + explicit edge keys/ids
      - Does NOT require kernel_info.txt
    """
#    _require_pythonhashseed()

    path_abs = str(Path(path).resolve())

    # 1) Compile -> .ll (clang_script.sh)
    cmd = ["/bin/bash", f"{get_root_path()}/src/clang_script.sh", str(name), str(path_abs), str(gg.type_graph)]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
    out, err = p.communicate()
#    print("returncode:", p.returncode)
#    print("stdout:\n", out)
#    print("stderr:\n", err)

    if p.returncode != 0:
        raise RuntimeError(f"clang_script.sh failed\nstdout:\n{out}\nstderr:\n{err}")

    # 2) Ensure placeholders exist
    src_file = join(path_abs, f"{name}.{src_ext}")
    if not os.path.isfile(src_file):
        raise FileNotFoundError(f"Missing source file: {src_file}")
    
    placeholders_file = src_file

    # placeholders_file = join(path_abs, f"{name}_placeholders.{src_ext}")
    # if not os.path.isfile(placeholders_file):
    #     placeholder_lines = insert_placeholders(src_file)
    #     with open(placeholders_file, "w", encoding="utf-8") as f:
    #         f.writelines(placeholder_lines)

    # 3) LLVM -> NetworkX
    g_nx = gg.llvm_to_nx(join(path_abs, name))  # ProGraML -> NetworkX (original function)
    # Canonicalize + relabel early so new node ids start from a deterministic base
    g_nx = gg.canonicalize_graph(g_nx)
    g_nx = gg.relabel_nodes_canonically(g_nx, rounds=3)
    g_nx = gg.canonicalize_graph(g_nx)

    g_nx_nodes = g_nx.number_of_nodes()

    # 4) Loop mapping (LLVM icmp vs source loops)
    for_dict_llvm, for_count_llvm = gg.get_icmp(path_abs, name)
    for_dict_source, for_count_source = gg.get_pragmas_loops(path_abs, f"{name}", EXT=src_ext)

    if for_count_llvm != for_count_source:
        raise RuntimeError(
            f"for-loop count mismatch: llvm={for_count_llvm} vs source={for_count_source}. "
            "This must match for stable pragma attachment."
        )

    # 5) Add pragma nodes WITHOUT kernel_info.txt (deterministic)
    new_nodes, new_edges = create_pragma_nodes_no_kernel_info(
        g_nx=g_nx,
        g_nx_nodes=g_nx_nodes,
        for_dict_source=for_dict_source,
        for_dict_llvm=for_dict_llvm,
        placeholder_src_file=placeholders_file,
        log=log,
    )

    # IMPORTANT: add with explicit keys/ids deterministically (do not use add_to_graph / auto-keys)
    gg.add_nodes_and_edges_with_explicit_keys(g_nx, new_nodes, new_edges)

    # 6) Process graph to processed/original/<name>_processed_result.gexf (deterministic)
    gg.process_graph(name, g_nx, csv_dict=None)

    # 7) Pseudo-block nodes (connected) -> processed/extended-pseudo-block-connected/
    aux_dir = Path(get_root_path()) / f"{gg.type_graph}/processed/extended-pseudo-block-connected"
    aux_dir.mkdir(parents=True, exist_ok=True)
    gg.add_auxiliary_nodes(
        name=name,
        path=gg.processed_gexf_folder,
        processed_path=str(aux_dir),
        csv_dict=None,
        node_type="block",
        connected=True,
    )

    # 8) Hierarchy edges -> processed/extended-pseudo-block-connected-hierarchy/
    hier_dir = Path(get_root_path()) / f"{gg.type_graph}/processed/extended-pseudo-block-connected-hierarchy"
    hier_dir.mkdir(parents=True, exist_ok=True)

    for_blocks_info = gg.get_for_blocks_info(name, path_abs)
    gg.augment_graph_hierarchy(
        name=name,
        for_blocks_info=for_blocks_info,
        src_path=str(aux_dir),
        dst_path=str(hier_dir),
        csv_dict=None,
        node_type="block",
    )

    final_gexf = hier_dir / f"{name}_processed_result.gexf"
    if out_gexf:
        out_path = Path(out_gexf).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final_gexf, out_path)
        final_gexf = out_path

    # Path to the specific file in the hierarchy directory
    hier_file = hier_dir / f"{name}_processed_result.gexf"
    if hier_file.exists():
        hier_file.unlink()

    # Path to the specific file in the auxiliary directory
    aux_file = aux_dir / f"{name}_processed_result.gexf"
    if aux_file.exists():
        aux_file.unlink()
        
    # Path to the file in the 'original' processed folder (created by gg.process_graph)
    orig_file = Path(gg.processed_gexf_folder) / "original" / f"{name}_processed_result.gexf"
    if orig_file.exists():
        orig_file.unlink()

    # Read final
    g_final = nx.readwrite.gexf.read_gexf(str(final_gexf))
    return g_final


#if __name__ == "__main__":

#    parser = argparse.ArgumentParser(description="Convert CPP to GEXF")

#    parser.add_argument("--name", type=str)
#    parser.add_argument("--ext", type=str)

    # 3. Parse the arguments
#    args = parser.parse_args()

    # 4. Use args.name and args.ext in your function
#    g = cpp_to_gexf(
#        name=args.name,
#        path="LLM_predictions",
#        src_ext=args.ext,
#        out_gexf=f"LLM_predictions/{args.name}_connected_hierarchy.gexf",
#        log=False,
#    )

#    print(f"OK: nodes={g.number_of_nodes()} edges={g.number_of_edges()}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert CPP to GEXF")

    parser.add_argument("--name", type=str, required=True,
                        help="basename without extension, e.g. network_placeholders")
    parser.add_argument("--ext", type=str, required=True,
                        help="source extension, e.g. cpp")
    parser.add_argument("--path", type=str, required=True,
                        help="directory containing the source file and headers")
    parser.add_argument("--out_gexf", type=str, default="",
                        help="optional output .gexf path")

    args = parser.parse_args()

    out_gexf = args.out_gexf if args.out_gexf else f"{args.path}/{args.name}_connected_hierarchy.gexf"

    g = cpp_to_gexf(
        name=args.name,
        path=args.path,
        src_ext=args.ext,
        out_gexf=out_gexf,
        log=False,
    )

    print(f"OK: nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
    print(f"GEXF saved to: {out_gexf}")
