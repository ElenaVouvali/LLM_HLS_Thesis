import networkx as nx
import json
import shutil
from os.path import join, abspath, basename, exists, dirname, isfile
from subprocess import Popen, PIPE
from collections import OrderedDict, Counter
from copy import deepcopy
import ast
from pprint import pprint
from glob import iglob
import glob
import csv
import re
import google.protobuf
import programl

from utils import create_dir_if_not_exists, get_root_path, natural_keys
from insert_placeholders import insert_placeholders

PRAGMA_POSITION = {
    'PIPELINE': 0,
    'UNROLL': 1,
    'ARRAY_PARTITION': 2
}

ARRAY_SCOPE_NODE_TYPE = 104
ARRAY_SCOPE_TEXT = "array_scope"    # equivalent of icmp node for loop pragmas in array pragmas

# New edge flow for array-scope <-> array declaration/use nodes
ARRAY_SCOPE_EDGE_FLOW = 7

# Keep array scope compact and informative
ARRAY_SCOPE_MAX_TARGETS = 8
ARRAY_SCOPE_MAX_PER_BLOCK = 2

type_graph = 'harp'
processed_gexf_folder = join(get_root_path(), f'{type_graph}/processed')


class Node():
    def __init__(self, block, function, text, type_n, features = None):
        self.block : int = block
        self.function : int = function
        self.text : str = text
        self.type_n : int = type_n ## 0: instruction, 1: variable, 2: immediate, 4: pseudo block, 100: pragma, 104: array scope
        self.features : str = features ## contains full text

    def get_attr(self, after_process = True):
        '''
            args:
                after_process : True if nodes are added to existing GNN-DSE graphs
                                False for initial graph generation in GNN-DSE
        '''
        n_dict = {}
        n_dict['block'] = self.block
        n_dict['function'] = self.function
        n_dict['text'] = self.text
        n_dict['type'] = self.type_n
        if after_process:
            n_dict['full_text'] = self.features
        else:
            n_dict['features'] = {'full_text': [self.features]}


        return n_dict

class Edge():
    def __init__(self, src, dst, flow, position):
        self.src : int = src
        self.dst : int = dst
        self.flow : int = flow ## 0: control, 1: data, 2: call, 200: pragma, 4: pseudo node for block, 5: connections between pseudo nodes, 6: for loop hierarchy
        self.position : int = position

    def get_attr(self):
        e_dict = {}
        e_dict['flow'] = self.flow
        e_dict['position'] = self.position

        return e_dict



def _node_sort_key(node, data):
    # Use only fields that exist in your graphs and are stable.
    # Adjust this to your schema.
    return (
        int(data.get("function", -1)),
        int(data.get("block", -1)),
        int(data.get("type", -1)),
        str(data.get("text", "")),
        str(data.get("full_text", "")),
        str(node),  # tie-breaker
    )


def create_pseudo_node_block(block, function):
    return Node(block, function, text = 'pseudo_block', type_n = 4, features = 'auxiliary node for each block')

def create_array_scope_node(block, function, varname, pragma_line):
    feat = f"array_scope<{varname}> from pragma: {pragma_line}"
    return Node(
        block=block,
        function=function,
        text=ARRAY_SCOPE_TEXT,
        type_n=ARRAY_SCOPE_NODE_TYPE,
        features=feat,
    )

def add_to_graph(g_nx, nodes, edges):
    if len(nodes) > 0:
        g_nx.add_nodes_from(nodes)
    if len(edges) > 0:
        g_nx.add_edges_from(edges)


def llvm_to_nx(name):
    '''
        reads a LLVM IR and converts it to a netwrokx graph

        args:
            name: name of the LLVM file/ kernel's name

        returns:
            g_nx: graph in networkx format
    '''
    filename = name + '.ll'
    with open(filename) as f:
        ll_file = f.read()
        G=programl.from_llvm_ir(ll_file)
        g_nx=programl.to_networkx(G)

    return g_nx


def extract_function_names(c_code):
    """
    Return list of (function_name, line_number) for top-level function definitions.
    We strip comments and ignore matches where the 'function name' is actually
    a control keyword such as 'if', 'for', 'while', etc.
    """

    # Strip C/C++ style comments but preserve line breaks where possible
    def _strip_comments(text):
        # Replace block comments with the same number of newlines so that line numbers remain roughly correct.
        def repl_block(m):
            return "\n" * m.group(0).count("\n")

        # Remove block comments: /* ... */
        text = re.sub(r"/\*.*?\*/", repl_block, text, flags=re.DOTALL)
        # Remove single-line comments: // ...
        text = re.sub(r"//.*", "", text)
        return text

    c_code_nocomments = _strip_comments(c_code)

    # Remove `extern "C" {` but keep newlines
    c_code_clean = re.sub(r'extern\s+"C"\s*{', '', c_code_nocomments)

    # pattern: <ret> <name>( ... ) {
    pattern = r'\b\w[\w\s\*]*\b([A-Za-z_]\w*)\s*\([^;]*\)\s*\{'

    # Do not treat control keywords as function names
    CONTROL_KEYWORDS = {"if", "for", "while", "switch", "else", "do", "case"}

    function_names = []
    for match in re.finditer(pattern, c_code_clean, flags=re.DOTALL):
        func_name = match.group(1)

        # Skip control constructs like "else if (...) {" being misread as a function
        if func_name in CONTROL_KEYWORDS:
            continue

        line_number = c_code_clean.count('\n', 0, match.start()) + 1
        function_names.append((func_name, line_number))

    return function_names



def get_icmp(path, name, log=False):
    llvm_file = join(path, f"{name}.ll")
    with open(llvm_file, "r") as f_llvm:
        lines_llvm = f_llvm.readlines()

    for_dict_llvm = OrderedDict()
#    for_dict_llvm = {}
    for_count_llvm = 0
    current_func = None
    local_for_count = 0

    for idx, line in enumerate(lines_llvm):
        s = line.strip()

        # Start of function
        if s.startswith("define"):
            current_func = s
#            for_dict_llvm[current_func] = OrderedDict()
            for_dict_llvm[current_func] = {}
            local_for_count = 0
            continue

        # for-loop header
        if s.startswith("for.cond") and current_func is not None:
            local_for_count += 1
            for_count_llvm += 1
            loop_id = local_for_count

            # Look ahead for first icmp before next basic block label
            for idx2, line2 in enumerate(lines_llvm[idx+1:], start=idx+1):
                t = line2.strip()
                if re.match(r'^[A-Za-z0-9_.]+:\s*$', t):
                    break
                if "icmp" in t:
                    for_dict_llvm[current_func][loop_id] = [t, idx, idx2]
                    break

    if log:
        sorted_log = {k: for_dict_llvm[k] for k in sorted(for_dict_llvm.keys())}
        print(json.dumps(sorted_log, indent=2))

    return for_dict_llvm, for_count_llvm



def get_pragmas_loops(path, name, EXT='cpp', top_func=None, log=False):
    '''
        gets a c/cpp kernel and returns the pragmas of each for loop

        args:
            path: parent directory of the kernel file
            name: kernel name (WITHOUT extension)

        returns:
            a dictionary with each entry showing the for loop and its pragmas
                {function_name: {for_loop_id: [for loop line, [list of pragmas]]}}
            number of for loops (total over all functions)
    '''

    src_file = join(path, f'{name}.{EXT}')

    with open(src_file, 'r') as f_source:
        code = f_source.read()

    lines_source = code.splitlines(True)

#    function_names_list = extract_function_names(code)
    function_names_list = sorted(extract_function_names(code), key=lambda x: x[1])

    if not function_names_list:
        # treat the whole file as a single "GLOBAL" function
        print(
            f"[WARN] extract_function_names() did not find any functions in {src_file}; "
            "treating whole file as GLOBAL."
        )
        function_names_list = [("GLOBAL", 1)]

#    for_dict_source = OrderedDict()
    for_dict_source = {}
    for_count_source = 0

    #  Structured scan (per function)
    for f_id, (f_name, start_line_1based) in enumerate(function_names_list):
        # convert to 0-based indices for lines_source
        start_idx = max(0, start_line_1based - 1)

        if f_id + 1 < len(function_names_list):
            next_start_line_1based = function_names_list[f_id + 1][1]
            end_idx = max(0, next_start_line_1based - 1)  # exclusive
        else:
            end_idx = len(lines_source)

        for_dict_source[f_name] = OrderedDict()
        local_for_count_source = 0

        idx = start_idx
        while idx < end_idx:
            line = lines_source[idx].strip()

            if not line or 'scop' in line:
                idx += 1
                continue

            # detect for loop
            if re.search(r'\bfor\s*\(', line):
                for_count_source += 1
                local_for_count_source += 1
                loop_line = line.strip('{')

                # collect pragmas after this for loop
                pragma_list = []
                j = idx + 1
                while j < end_idx:
                    next_line = lines_source[j].strip()
                    # skip labels like "L1:"
                    if next_line.endswith(":") and not next_line.startswith("#pragma"):
                        j += 1
                        continue
                    if next_line.startswith("#pragma") and "KERNEL" not in next_line.upper():
                        pragma_list.append(next_line)
                        j += 1
                    else:
                        break

                for_dict_source[f_name][local_for_count_source] = [loop_line, pragma_list]
                idx = j
                continue

            idx += 1

    # if we found 0 for-loops, do a flat scan
    if for_count_source == 0:
        print(f"[WARN] No for-loops found in structured scan of {src_file}; "
              f"applying flat fallback scan.")

        fallback_loops = []
        for idx, line in enumerate(lines_source):
            if re.search(r'\bfor\s*\(', line):
                fallback_loops.append((idx, line.strip()))

        if fallback_loops:
            if top_func and top_func.strip():
                fname = top_func.strip()
            elif function_names_list:
                fname = function_names_list[0][0]
            else:
                fname = "FALLBACK"
            
            for_dict_source = {fname: {}}
#            for_dict_source = OrderedDict()
#            for_dict_source[fname] = OrderedDict()

            for local_id, (lineno, loop_line) in enumerate(fallback_loops, start=1):
                for_dict_source[fname][local_id] = [loop_line, []]  # no pragmas
            for_count_source = len(fallback_loops)

    sorted_final_dict = OrderedDict()
    for f_name in sorted(for_dict_source.keys()):
        # Also ensure loop IDs are sorted (1, 2, 3...)
        sorted_loops = OrderedDict()
        for l_id in sorted(for_dict_source[f_name].keys()):
            sorted_loops[l_id] = for_dict_source[f_name][l_id]
        sorted_final_dict[f_name] = sorted_loops

    if log:
        print(json.dumps(sorted_final_dict, indent=4))

    return sorted_final_dict, for_count_source


def get_pragmas_arrays(placeholder_cpp_file, log=False):
    """
    Parse *_placeholders.c/.cpp and collect all
    `#pragma HLS array_partition` pragmas.

    Returns a list of dicts:
        {
          "function": <source function name or None>,
          "var": <array variable name>,
          "pragma": <full pragma line>,
          "type": <partition type or None>,
          "factor": <factor token or None>,
          "dim": <dim token or None>,
        }
    """
    with open(placeholder_cpp_file, 'r') as f:
        lines = f.readlines()

    function_names_list = extract_function_names(''.join(lines))
    func_spans = []
    for i, (fname, start_line) in enumerate(function_names_list):
        start_idx = start_line - 1
        if i + 1 < len(function_names_list):
            end_idx = function_names_list[i + 1][1] - 2
        else:
            end_idx = len(lines) - 1
        func_spans.append((fname, start_idx, end_idx))

    line_to_func = {}
    for fname, s, e in func_spans:
        for li in range(s, e + 1):
            line_to_func[li] = fname

    array_pragmas = []
    for idx, line in enumerate(lines):
        s = line.strip()
        s_low = s.lower()

        if not s_low.startswith("#pragma"):
            continue
        if "array_partition" not in s_low:
            continue

        m_var = re.search(r'\bvariable\s*=\s*([A-Za-z_]\w*)\b', s, flags=re.IGNORECASE)
        if not m_var:
            continue

        m_type   = re.search(r'\btype\s*=\s*([A-Za-z_]\w*)\b',   s, flags=re.IGNORECASE)
        m_factor = re.search(r'\bfactor\s*=\s*([^\s]+)',         s, flags=re.IGNORECASE)
        m_dim    = re.search(r'\bdim\s*=\s*([^\s]+)',            s, flags=re.IGNORECASE)

        array_pragmas.append({
            "function": line_to_func.get(idx, None),
            "var": m_var.group(1),
            "pragma": s,
            "type":   m_type.group(1) if m_type else None,
            "factor": m_factor.group(1) if m_factor else None,
            "dim":    m_dim.group(1) if m_dim else None,
        })

    if log:
        pprint(array_pragmas)

    return array_pragmas



LABEL_IN_COMMENT = re.compile(r'/\*\s*(L\d+)\s*:\s*\*/', re.IGNORECASE)
LABEL_AT_START   = re.compile(r'^\s*(L\d+)\s*:', re.IGNORECASE)

def extract_loop_label(loop_line: str):
    if not isinstance(loop_line, str):
        return None
    m = LABEL_IN_COMMENT.search(loop_line)
    if m:
        return m.group(1).upper()
    m = LABEL_AT_START.match(loop_line)
    if m:
        return m.group(1).upper()
    return None


def load_tripcounts_by_label(kernel_info_file):
    by_label = {}  # 'L6' -> 2048
    with open(kernel_info_file) as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = [p.strip() for p in line.split(',')]
            # Expect: <Label>,loop,<bound>,<LoopName>
            if len(parts) >= 3 and parts[1].lower() == 'loop':
                label = parts[0].upper()
                bound = int(parts[2])
                by_label[label] = bound
    return by_label


def check_tripcount_consistency(for_loop_text, icmp_inst, tripcounts_by_label):
    """
    Compare loop info from kernel_info (by label) with the info from the LLVM icmp.
    Returns the label (or None) so that the caller can still use it for attaching pragmas.
    """
    label = extract_loop_label(for_loop_text)
    if not label or label not in tripcounts_by_label:
        return label  # nothing to check or no entry in kernel_info

    TC_for = tripcounts_by_label[label]  # value from kernel_info.txt

    # Strip debug info
    icmp_inst_ = icmp_inst.split('!dbg')[0]

    # Try to parse predicate and bound from the icmp
    m_pred = re.search(r'icmp\s+(\w+)\s+i\d+\s+[^,]+,\s*([-0-9]+)', icmp_inst_)
    if not m_pred:
        return label

    pred = m_pred.group(1)
    try:
        bound_icmp = int(m_pred.group(2))  # the constant in the compare
    except ValueError:
        # Non-integer bound --> skip check
        return label

    m_init = re.search(r'for\s*\(\s*[^=]+=\s*([0-9]+)', for_loop_text)
    start = int(m_init.group(1)) if m_init else None

    if start is None or pred in {"sge", "sgt"}:
        return label

    # We consider two possible interpretations for the kernel_info value:
    #  1) It may store the **bound** (the constant in the compare).
    #  2) It may store the **tripcount** (number of iterations).
    # If either fits (or is off by 1), we treat it as consistent.

    # 'bound semantics' – kernel_info value == bound in icmp
    bound_ok = (TC_for == bound_icmp)

    # 'tripcount semantics' – derive tripcount from bound + start
    tripcount_icmp = None
    if pred in {"slt", "ult"}:
        tripcount_icmp = max(0, bound_icmp - start)
    elif pred in {"sle", "ule"}:
        tripcount_icmp = max(0, bound_icmp - start + 1)

    trip_ok = (tripcount_icmp is not None and TC_for == tripcount_icmp)

    # Soft tolerance: ignore off-by-one between the two interpretations
    if not bound_ok and not trip_ok:
        if tripcount_icmp is not None:
            if (abs(TC_for - tripcount_icmp) <= 1) or (abs(TC_for - bound_icmp) <= 1):
                return label  # close enough, do not warn

        # Otherwise keep the warning
        print(
            f"[WARN] Tripcount mismatch for loop {for_loop_text} "
            f"label={label}: TC_for={TC_for}, "
            f"icmp_bound={bound_icmp}, start={start}, "
            f"trip_icmp={tripcount_icmp}"
        )

    return label


def _get_node_full_text(ndata):
    if "full_text" in ndata and ndata["full_text"] is not None:
        return str(ndata["full_text"])

    if "features" in ndata:
        feat = ndata["features"]
        try:
            if isinstance(feat, str):
                feat = ast.literal_eval(feat)
            if isinstance(feat, dict):
                ft = feat.get("full_text", [None])[0]
                if ft is not None:
                    return str(ft)
        except Exception:
            pass

    return None


def _parse_llvm_function_bodies(kernel_info_file: str):
    """
    Infer which numeric graph function-id corresponds to a given LLVM function definition string.
    """
    src_dir = os.path.dirname(kernel_info_file)
    kernel_name = os.path.basename(src_dir)
    llvm_file = os.path.join(src_dir, f"{kernel_name}.ll")

    out = {}
    current_func = None

    with open(llvm_file, "r") as f:
        for line in f:
            s = line.strip()

            if s.startswith("define"):
                current_func = s
                out[current_func] = []
                continue

            if current_func is None:
                continue

            if s == "}":
                current_func = None
                continue

            if not s or s.startswith(";"):
                continue

            # skip basic-block labels like "for.cond:" / "entry:"
            if re.match(r'^[A-Za-z0-9_.]+:\s*$', s):
                continue

            out[current_func].append(s.split('!dbg')[0].strip())

    return out


def infer_graph_function_id(g_nx, llvm_key: str, llvm_func_bodies: dict):
    """
    Infer which numeric graph function-id corresponds to llvm_key by voting over
    exact instruction-text matches.
    """
    counts = Counter()

    for inst in llvm_func_bodies.get(llvm_key, []):
        matched_func_ids = set()

        for _, ndata in g_nx.nodes(data=True):
            full_text = _get_node_full_text(ndata)
            if not isinstance(full_text, str):
                continue

            full_text = full_text.split('!dbg')[0].strip()
            if full_text == inst:
                fid = int(ndata.get("function", -1))
                if fid != -1:
                    matched_func_ids.add(fid)

        if len(matched_func_ids) == 1:
            counts[next(iter(matched_func_ids))] += 5
        else:
            for fid in matched_func_ids:
                counts[fid] += 1

    if not counts:
        return None

    # deterministic winner
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _compile_var_token_regexes(varname: str):
    """
    Conservative LLVM-aware token regexes.
    We intentionally support: %A, @A, A but reject accidental substring matches (A does not match AA)
    """
    ident_chars = r'A-Za-z0-9_.$'
    v = re.escape(varname)

    return [
        re.compile(rf'(?<![{ident_chars}])%{v}(?![{ident_chars}])'),
        re.compile(rf'(?<![{ident_chars}])@{v}(?![{ident_chars}])'),
        re.compile(rf'(?<![{ident_chars}]){v}(?![{ident_chars}])'),
    ]


def _node_mentions_var(full_text: str, var_regexes) -> bool:
    if not isinstance(full_text, str):
        return False
    return any(rx.search(full_text) for rx in var_regexes)


def _is_array_decl_candidate(full_text: str, varname: str) -> bool:
    """
    Prefer exact LLVM allocation sites such as:
        %A = alloca [1024 x i32], ...
    """
    if not isinstance(full_text, str):
        return False

    ident_chars = r'A-Za-z0-9_.$'
    v = re.escape(varname)
    decl_re = re.compile(rf'(?<![{ident_chars}])%{v}\s*=\s*alloca\b')
    return decl_re.search(full_text) is not None


def _array_target_priority(node, ndata):
    """
    Lower tuple = better representative target for array scope.
    Prefer declaration/access-related nodes over generic mentions.
    """
    ft = _get_node_full_text(ndata)
    ft = "" if ft is None else ft.split('!dbg')[0].strip()

    if "getelementptr" in ft:
        kind_rank = 0
    elif re.search(r'\b(load|store)\b', ft):
        kind_rank = 1
    elif "call" in ft:
        kind_rank = 2
    elif int(ndata.get("type", -1)) == 1:
        kind_rank = 3
    else:
        kind_rank = 4

    return (
        kind_rank,
        int(ndata.get("block", -1)),
        int(ndata.get("type", -1)),
        str(ndata.get("text", "")),
        ft,
        str(node),
    )


def _is_immediate_node(ndata):
    return int(ndata.get("type", -1)) == 2

def _is_call_target(ndata):
    ft = _get_node_full_text(ndata)
    if not isinstance(ft, str):
        return False
    ft = ft.split('!dbg')[0].strip()
    return "call" in ft


def _call_callee_name(ndata):
    ft = _get_node_full_text(ndata)
    if not isinstance(ft, str):
        return ""
    ft = ft.split('!dbg')[0].strip()
    m = re.search(r'\bcall\b.*?@([^(]+)\(', ft)
    return m.group(1) if m else ft


def _iter_same_func_flow_neighbors(
    g_nx,
    node,
    expected_graph_func_id,
    allowed_flows=(1,),
    skip_types=(2,),
):
    seen = set()

    for u, _, _, ed in g_nx.in_edges(node, keys=True, data=True):
        if int(ed.get("flow", -1)) not in allowed_flows:
            continue
        nd = g_nx.nodes[u]
        if int(nd.get("function", -1)) != int(expected_graph_func_id):
            continue
        if int(nd.get("type", -1)) in skip_types:
            continue
        if u not in seen:
            seen.add(u)
            yield u, nd

    for _, v, _, ed in g_nx.out_edges(node, keys=True, data=True):
        if int(ed.get("flow", -1)) not in allowed_flows:
            continue
        nd = g_nx.nodes[v]
        if int(nd.get("function", -1)) != int(expected_graph_func_id):
            continue
        if int(nd.get("type", -1)) in skip_types:
            continue
        if v not in seen:
            seen.add(v)
            yield v, nd


def _is_array_semantic_bridge(ndata, var_regexes):
    ft = _get_node_full_text(ndata)
    if not isinstance(ft, str):
        return False
    ft = ft.split('!dbg')[0].strip()

    if _node_mentions_var(ft, var_regexes):
        return True

    return any(tok in ft for tok in (
        "getelementptr",
        "bitcast",
        "load",
        "store",
    ))


def _collect_array_call_relays(
    g_nx,
    seed_nodes,
    expected_graph_func_id,
    var_regexes,
    max_hops=2,
):
    """
    Find caller-side call relays, but only through array-relevant paths.
    Reject spurious paths through immediates like i64 0.
    """
    queue = []
    visited = set()
    relays = []

    for node, ndata in seed_nodes:
        queue.append((node, 0, True))   # seed nodes are already array-relevant
        visited.add((node, True))

    while queue:
        cur, dist, is_array_relevant = queue.pop(0)
        if dist >= max_hops:
            continue

        for nb, nd in _iter_same_func_flow_neighbors(
            g_nx,
            cur,
            expected_graph_func_id,
            allowed_flows=(1,),
            skip_types=(2,),   # skip immediates
        ):
            next_relevant = is_array_relevant or _is_array_semantic_bridge(nd, var_regexes)

            if _is_call_target(nd):
                # Only keep the call if we reached it through an array-relevant path
                if is_array_relevant:
                    relays.append((nb, nd, dist + 1))
                continue

            state = (nb, next_relevant)
            if state in visited:
                continue
            visited.add(state)
            queue.append((nb, dist + 1, next_relevant))

    relays = sorted(relays, key=lambda x: (x[2], _array_target_priority(x[0], x[1])))

    out = []
    seen = set()
    for n, d, _ in relays:
        if n not in seen:
            seen.add(n)
            out.append((n, d))
    return out


def _select_array_scope_targets(
    matched_nodes,
    decl_candidates,
    relay_nodes=None,
    max_total=ARRAY_SCOPE_MAX_TARGETS,
    max_per_block=ARRAY_SCOPE_MAX_PER_BLOCK,
):
    matched_nodes = sorted(matched_nodes, key=lambda nd: _node_sort_key(nd[0], nd[1]))
    decl_candidates = sorted(decl_candidates, key=lambda nd: _node_sort_key(nd[0], nd[1]))
    relay_nodes = [] if relay_nodes is None else sorted(
        relay_nodes, key=lambda nd: _array_target_priority(nd[0], nd[1])
    )

    selected = []
    seen = set()

    # 1) declaration first
    if decl_candidates:
        node0, data0 = decl_candidates[0]
        selected.append((node0, data0))
        seen.add(node0)

    # 2) reserve up to two distinct caller-side call relays if present
    kept_callees = set()
    for node0, data0 in relay_nodes:
        callee = _call_callee_name(data0)
        if node0 in seen:
            continue
        if callee in kept_callees:
            continue

        selected.append((node0, data0))
        seen.add(node0)
        kept_callees.add(callee)

        if len(kept_callees) >= 2 or len(selected) >= max_total:
            break

    # 3) preserve per-block coverage
    by_block = {}
    for node, ndata in matched_nodes:
        if node in seen:
            continue
        block = int(ndata.get("block", -1))
        by_block.setdefault(block, []).append((node, ndata))

    for block in sorted(by_block.keys()):
        ranked = sorted(by_block[block], key=lambda nd: _array_target_priority(nd[0], nd[1]))
        taken = 0
        for node, ndata in ranked:
            if node in seen:
                continue
            selected.append((node, ndata))
            seen.add(node)
            taken += 1
            if len(selected) >= max_total or taken >= max_per_block:
                break
        if len(selected) >= max_total:
            break

    # 4) fill remaining budget globally
    if len(selected) < max_total:
        leftovers = []
        for block in sorted(by_block.keys()):
            leftovers.extend(by_block[block])
        leftovers = sorted(leftovers, key=lambda nd: _array_target_priority(nd[0], nd[1]))

        for node, ndata in leftovers:
            if node in seen:
                continue
            selected.append((node, ndata))
            seen.add(node)
            if len(selected) >= max_total:
                break

    return selected


def create_pragma_nodes(g_nx, g_nx_nodes, kernel_info_file, for_dict_source, for_dict_llvm, log=False):
    """
    Create pragma nodes (loop pragmas + array pragmas) and edges.
    - Loop pragmas are attached to the LLVM icmp node of the corresponding loop.
    - Array pragmas are attached to LLVM nodes that reference the correct array in the correct function, using token-aware matching.
    """
    new_nodes, new_edges = [], []
    next_node_id = g_nx_nodes

    tripcounts_by_label = load_tripcounts_by_label(kernel_info_file)
    eligible_labels = set(tripcounts_by_label.keys())
    llvm_func_bodies = _parse_llvm_function_bodies(kernel_info_file)

    def resolve_llvm_key(src_func_name: str):
        """
        Find the LLVM function definition that corresponds to src_func_name.
        """
        exact_matches = []
        suffix_matches = []
        substring_matches = []

        for key in sorted(for_dict_llvm.keys()):
            m = re.search(r'@([^(]+)\s*\(', key)
            if not m:
                continue
            mangled = m.group(1)

            demangled = None
            m2 = re.match(r'_Z(\d+)([A-Za-z_]\w*)', mangled)
            if m2:
                try:
                    name_len = int(m2.group(1))
                    candidate = m2.group(2)
                    if len(candidate) == name_len:
                        demangled = candidate
                except ValueError:
                    pass

            if demangled is None:
                demangled = mangled

            if demangled == src_func_name or mangled == src_func_name:
                exact_matches.append(key)
                continue

            if demangled.endswith(src_func_name) or mangled.endswith(src_func_name):
                suffix_matches.append(key)
                continue

            if src_func_name in mangled or src_func_name in demangled:
                substring_matches.append(key)

        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            print(f"[WARN] Multiple LLVM functions matched source function '{src_func_name}' (exact): {exact_matches}")
            return None

        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            print(f"[WARN] Multiple LLVM functions matched source function '{src_func_name}' (suffix): {suffix_matches}")
            return None

        if len(substring_matches) == 1:
            return substring_matches[0]
        if len(substring_matches) > 1:
            print(f"[WARN] Multiple LLVM functions matched source function '{src_func_name}' (substring): {substring_matches}")
            return None

        if len(for_dict_llvm) == 1:
            only_key = next(iter(for_dict_llvm.keys()))
            print(f"[WARN] No LLVM function matched source function '{src_func_name}'. Falling back to sole LLVM function '{only_key}'.")
            return only_key

        print(f"[WARN] Could not match source function '{src_func_name}' to any LLVM function. Available: {list(for_dict_llvm.keys())}")
        return None

    def find_icmp_node(icmp_inst: str, expected_graph_func_id=None):
        """
        Find the graph node whose full_text exactly matches icmp_inst,
        optionally restricting to a single graph function-id.
        """
        target = icmp_inst.split('!dbg')[0].strip()
        matches = []

        for node, ndata in g_nx.nodes(data=True):
            if expected_graph_func_id is not None:
                if int(ndata.get("function", -1)) != int(expected_graph_func_id):
                    continue

            full_text = _get_node_full_text(ndata)
            if not isinstance(full_text, str):
                continue

            full_text = full_text.split('!dbg')[0].strip()
            if full_text == target:
                matches.append((node, ndata))

        if not matches:
            return None, None, None

        matches = sorted(matches, key=lambda nd: _node_sort_key(nd[0], nd[1]))
        if len(matches) > 1 and log:
            print(
                f"[WARN] Multiple icmp-node matches for '{target}' "
                f"in function {expected_graph_func_id}; choosing the first deterministically."
            )

        node0, data0 = matches[0]
        block_id = int(data0.get("block", -1))
        func_id = int(data0.get("function", -1))
        return int(node0), block_id, func_id

    graph_func_ids = sorted({
        int(ndata.get("function", -1))
        for _, ndata in g_nx.nodes(data=True)
        if int(ndata.get("function", -1)) != -1
    })

    # --------------------------------------------------
    # LOOP PRAGMAS
    # --------------------------------------------------
    for f_name in sorted(for_dict_source.keys()):
        f_content = for_dict_source[f_name]
        if not f_content:
            continue

        llvm_key = resolve_llvm_key(f_name)
        if llvm_key is None:
            continue

        llvm_content = for_dict_llvm[llvm_key]
        expected_graph_func_id = infer_graph_function_id(g_nx, llvm_key, llvm_func_bodies)
        if expected_graph_func_id is None:
            raise RuntimeError(
                f"Could not infer graph function-id for loop source function '{f_name}' "
                f"(LLVM key: {llvm_key}). Refusing global fallback."
            )

        for for_loop_id in sorted(f_content.keys()):
            payload = f_content[for_loop_id]

            if isinstance(payload, dict):
                for_loop_text = payload["loop_line"]
                pragmas = payload["pragmas"]
                local_id = payload.get("local_id", for_loop_id)
            else:
                for_loop_text, pragmas = payload
                local_id = for_loop_id

            if local_id not in llvm_content:
                print(
                    f"[WARN] LLVM loop id {local_id} not found for function '{f_name}' "
                    f"(available: {list(llvm_content.keys())}). Skipping this loop."
                )
                continue

            icmp_inst = llvm_content[local_id][0]
            label = check_tripcount_consistency(for_loop_text, icmp_inst, tripcounts_by_label)

            if label and label not in eligible_labels:
                continue

            node_id, block_id, function_id = find_icmp_node(
                icmp_inst,
                expected_graph_func_id=expected_graph_func_id,
            )
            if node_id is None:
                print(f"[WARN] icmp instruction not found in graph: {icmp_inst}")
                continue

            for pragma in pragmas:
                tokens = pragma.split()
                if len(tokens) < 3:
                    print(f"[WARN] Unexpected pragma format (too few tokens): {pragma}")
                    continue

                pragma_kind = tokens[2].upper()
                if pragma_kind not in PRAGMA_POSITION:
                    print(f"[WARN] Skipping unknown pragma kind: {pragma_kind}")
                    continue

                p_dict = {
                    "type": 100,
                    "block": block_id,
                    "function": function_id,
                    "features": {"full_text": [pragma]},
                    "text": pragma_kind,
                }
                new_nodes.append((next_node_id, p_dict))

                e_attr = {"flow": 200, "position": PRAGMA_POSITION[pragma_kind]}
                new_edges.append((node_id, next_node_id, e_attr))
                new_edges.append((next_node_id, node_id, e_attr))

                next_node_id += 1

    # --------------------------------------------------
    # ARRAY PRAGMAS
    # --------------------------------------------------
    src_dir = os.path.dirname(kernel_info_file)
    kernel_name = os.path.basename(src_dir)

    placeholder_src = None
    for ext in (".cpp", ".c"):
        candidate = os.path.join(src_dir, f"{kernel_name}_placeholders{ext}")
        if os.path.isfile(candidate):
            placeholder_src = candidate
            break

    if placeholder_src:
        array_pragmas = get_pragmas_arrays(placeholder_src, log=log)
        array_pragmas = sorted(array_pragmas, key=lambda x: (x["function"] or "", x["var"], x["pragma"]))

        for ap in array_pragmas:
            src_func_name = ap["function"]
            varname = ap["var"]
            pragma_line = ap["pragma"]

            # Map source function -> LLVM function -> numeric graph function-id
            if src_func_name is None:
                if len(graph_func_ids) == 1:
                    expected_graph_func_id = graph_func_ids[0]
                else:
                    print(
                        f"[WARN] Array pragma for variable '{varname}' has no source-function "
                        f"and graph has multiple functions {graph_func_ids}. Skipping.\n"
                        f"  pragma: {pragma_line}"
                    )
                    continue
            else:
                llvm_key = resolve_llvm_key(src_func_name)
                if llvm_key is None:
                    print(
                        f"[WARN] Could not resolve LLVM function for array pragma variable '{varname}' "
                        f"in source function '{src_func_name}'. Skipping."
                    )
                    continue

                expected_graph_func_id = infer_graph_function_id(g_nx, llvm_key, llvm_func_bodies)
                if expected_graph_func_id is None:
                    raise RuntimeError(
                        f"Could not infer graph function-id for array pragma variable '{varname}' "
                        f"in source function '{src_func_name}' (LLVM key: {llvm_key})."
                    )

            var_regexes = _compile_var_token_regexes(varname)

            matched_nodes = []
            decl_candidates = []

            for node, ndata in g_nx.nodes(data=True):
                if int(ndata.get("function", -1)) != int(expected_graph_func_id):
                    continue

                ntype = int(ndata.get("type", -1))
                if ntype not in (0, 1):
                    continue

                full_text = _get_node_full_text(ndata)
                if not isinstance(full_text, str):
                    continue

                ft_str = full_text.split('!dbg')[0].strip()
                if not _node_mentions_var(ft_str, var_regexes):
                    continue

                matched_nodes.append((node, ndata))

                if _is_array_decl_candidate(ft_str, varname):
                    decl_candidates.append((node, ndata))

            if not matched_nodes:
                if log:
                    print(
                        f"[WARN] No graph node found for array '{varname}' "
                        f"in function-id {expected_graph_func_id} "
                        f"(pragma: {pragma_line})"
                    )
                continue

            matched_nodes = sorted(matched_nodes, key=lambda nd: _node_sort_key(nd[0], nd[1]))
            seen = set()
            matched_nodes = [(n, d) for (n, d) in matched_nodes if not (n in seen or seen.add(n))]

            decl_candidates = sorted(decl_candidates, key=lambda nd: _node_sort_key(nd[0], nd[1]))

            # canonical home of the array scope = declaration block if available
            if decl_candidates:
                anchor_node, anchor_data = decl_candidates[0]
            else:
                anchor_node, anchor_data = matched_nodes[0]

            block_id = int(anchor_data.get("block", -1))
            function_id = int(anchor_data.get("function", -1))

            # 1) create pragma node
            pragma_node_id = next_node_id
            next_node_id += 1

            pragma_dict = {
                "type": 100,
                "block": block_id,
                "function": function_id,
                "features": {"full_text": [pragma_line]},
                "text": "ARRAY_PARTITION",
            }
            new_nodes.append((pragma_node_id, pragma_dict))

            # 2) create array-scope node
            scope_node_id = next_node_id
            next_node_id += 1

            scope_node = create_array_scope_node(
                block=block_id,
                function=function_id,
                varname=varname,
                pragma_line=pragma_line,
            )
            scope_dict = scope_node.get_attr(after_process=False)
            scope_dict["array_var"] = varname
            new_nodes.append((scope_node_id, scope_dict))

            # 3) pragma <-> array-scope (same semantic role as loop pragma <-> icmp)
            pragma_edge_attr = {
                "flow": 200,
                "position": PRAGMA_POSITION["ARRAY_PARTITION"],
            }
            new_edges.append((pragma_node_id, scope_node_id, pragma_edge_attr))
            new_edges.append((scope_node_id, pragma_node_id, pragma_edge_attr))

            # 4) array-scope <-> representative declaration/use targets
            relay_nodes = _collect_array_call_relays(
                g_nx,
                matched_nodes,
                expected_graph_func_id=expected_graph_func_id,
                var_regexes=var_regexes,
                max_hops=2,
            )

            selected_targets = _select_array_scope_targets(
                matched_nodes=matched_nodes,
                decl_candidates=decl_candidates,
                relay_nodes=relay_nodes,
            )

            for pos, (node, _) in enumerate(selected_targets):
                scope_edge_attr = {
                    "flow": ARRAY_SCOPE_EDGE_FLOW,
                    "position": pos,
                }
                new_edges.append((scope_node_id, node, scope_edge_attr))
                new_edges.append((node, scope_node_id, scope_edge_attr))

            if log:
                touched_blocks = sorted({int(nd.get("block", -1)) for _, nd in selected_targets})
                print(
                    f"[INFO] ARRAY scope created: var={varname} "
                    f"src_func={src_func_name} graph_func={function_id} "
                    f"decl_found={bool(decl_candidates)} "
                    f"relay_found={bool(relay_nodes)} "
                    f"selected_targets={len(selected_targets)} "
                    f"blocks={touched_blocks}"
                )

    if log:
        pprint(new_nodes)
        pprint(new_edges)

    return new_nodes, new_edges



def prune_redundant_nodes(g_new):
    while True:
        nodes_to_check = sorted(g_new.nodes())
        remove_nodes = [n for n in nodes_to_check if g_new.degree(n) == 0 or n is None]
        #remove_nodes = set()
        #for node in g_new.nodes():
        #    if len(list(g_new.neighbors(node))) == 0 or node is None:
        #        print(node)
        #        remove_nodes.add(node)
        #for node in remove_nodes:
        #    g_new.remove_node(node)
        if not remove_nodes:
            break
        g_new.remove_nodes_from(remove_nodes)


def get_for_blocks_info(name, path):
    with open(join(path, f'{name}.ll'), 'r') as f_llvm:
        lines_llvm = f_llvm.readlines()

    for_blocks_info = OrderedDict() # label: {ind: loop number, preds:, next_instr:, line_num:, end: [(for.end line num, for.end label)], possible_children: children:}
    # possible_children are all children, children is only first level children
    # check up to 3 next instr to make sure the block is correct
    for_stack = [] # push for.cond pop for.end
    for_start = []
    for_end = []
    for_label = []
    i = 0
    correct_func = None
    for idx, line in enumerate(lines_llvm):
        s = line.strip()
        if s.startswith("define"):
            current_func = s
            continue
        if line.startswith('for.'):
            content = line.strip().split(';')
            line = content[0].strip()
            if 'for.cond' in line:
                key = f'{line}{idx}'
                assert key not in for_blocks_info
                for_blocks_info[key] = {
                    'ind': i,
                    'preds': content[1],
                    'next_instr': [lines_llvm[idx+1].strip(), lines_llvm[idx+2].strip(), lines_llvm[idx+3].strip()],
                    'line_num': idx,
                    'llvm_func': current_func,
                }
                for_stack.append(key)
                i += 1
            elif 'for.end' in line:
                res_cond = for_stack.pop()
                assert res_cond in for_blocks_info
                for_blocks_info[res_cond]['end'] = (idx, line)

    for for_l, for_l_value in for_blocks_info.items():
        if 'cond' in for_l:
            for_start.append(for_l_value['line_num'])
            for_end.append(for_l_value['end'][0])
            for_label.append(for_l)
    for idx, start_num in enumerate(for_start):
        child_idx = idx + 1
        possible_children = []
        for s, e in zip(for_start[idx+1:], for_end[idx+1:]):
            if s > start_num and e < for_end[idx]:
                possible_children.append(for_label[child_idx])
                child_idx += 1
            else:
                break
        for_blocks_info[for_label[idx]]['possible_children'] = possible_children

    for for_l, for_l_value in for_blocks_info.items():
        possible_children = for_l_value['possible_children']
        children = []
        i = 0
        while i < len(possible_children):
            children.append(possible_children[i])
            i += len(for_blocks_info[possible_children[i]]['possible_children']) + 1
        for_l_value['children'] = children

    return for_blocks_info


def load_kernel_source_map(csv_path):
    """
    Load a mapping from kernel directory name (app_name) -> metadata.

    Returns:
        mapping[app_name] = {
            "file_name": "<source file inside ApplicationDataset>",
            "ext": ".c" or ".cpp",
            "top": "<top_level_function>",
        }
    """
    mapping = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        required = {"app_name", "top_level_function", "file_name", "file_name_extension"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        for row in reader:
            app = row["app_name"].strip()
            fname = row["file_name"].strip()
            ext_col = row["file_name_extension"].strip()
            top = row["top_level_function"].strip()

            if not app or not fname:
                continue

            # Derive extension robustly
            base, ext = os.path.splitext(fname)
            if not ext and ext_col:
                ext = "." + ext_col
            elif ext and ext_col and ext.lstrip(".") != ext_col:
                # Optional: warn if CSV extension disagrees with file_name suffix
                # print(f"[WARN] Mismatch in extension for {app}: {fname} vs {ext_col}")
                pass

            if not ext:
                raise ValueError(f"Could not determine extension for app '{app}', file '{fname}'")

            mapping[app] = {
                "file_name": fname,
                "ext": ext,       # includes the dot, e.g. ".c" or ".cpp"
                "top": top,
            }

    return mapping


def write_csv_file(csv_dict, csv_header, file_path):
    with open(join(get_root_path(), file_path), mode = 'w') as f:
        f_writer = csv.DictWriter(f, fieldnames=csv_header)
        f_writer.writeheader()
        for d, value in csv_dict.items():
            if d == 'header':
                continue
            f_writer.writerow(value)


# =============================
# Deterministic hardening layer
# =============================

# graph_gen_deterministic.py
#
# Deterministic hardening layer (merged into this single file).
#
# How to run:
#   PYTHONHASHSEED=0 python -m graph_gen_deterministic
#
# Why PYTHONHASHSEED:
#   Python can randomize hashes across runs; setting PYTHONHASHSEED must occur
#   BEFORE interpreter startup to be effective. See CPython docs.  :contentReference[oaicite:4]{index=4}

import os
import json
import hashlib
from copy import deepcopy
from typing import Any, Dict, Tuple, List, Optional

import networkx as nx

# -----------------------------
# Hard determinism guardrails
# -----------------------------

def _require_pythonhashseed() -> None:
    seed = os.environ.get("PYTHONHASHSEED", "")
    if seed == "":
        raise RuntimeError(
            "Determinism requires PYTHONHASHSEED to be set before Python starts.\n"
            "Run like:\n"
            "  PYTHONHASHSEED=0 python -m graph_gen_deterministic\n"
        )

_require_pythonhashseed()


# -----------------------------
# Stable attribute helpers
# -----------------------------

def det_get_full_text(ndata: Dict[str, Any]) -> str:
    """
    Extract stable 'full_text' for sorting/matching across the pipeline.
    Handles both:
      - ProGraML style: ndata['features']={'full_text':[...]}
      - Post-processed: ndata['full_text']=...
    """
    if "full_text" in ndata and ndata["full_text"] is not None:
        return str(ndata["full_text"])

    feat = ndata.get("features")
    if isinstance(feat, dict):
        ft = feat.get("full_text")
        if isinstance(ft, list) and ft:
            return str(ft[0])

    # Some of your code stringifies features; try parsing conservatively
    if "features" in ndata and isinstance(ndata["features"], str):
        try:
            obj = eval(ndata["features"], {"__builtins__": {}})  # safer than full eval
            if isinstance(obj, dict):
                ft = obj.get("full_text")
                if isinstance(ft, list) and ft:
                    return str(ft[0])
        except Exception:
            pass

    return ""


def det_node_sort_key(node: Any, data: Dict[str, Any]) -> Tuple:
    """
    Stable node key based ONLY on stable attributes.
    Do NOT use the original node id as a tie-breaker if you want node-id invariance.
    """
    return (
        int(data.get("function", -1)),
        int(data.get("block", -1)),
        int(data.get("type", -1)),
        str(data.get("text", "")),
        det_get_full_text(data),
    )


def det_edge_sort_key(
    u: Any,
    v: Any,
    data: Dict[str, Any],
    node_rank: Dict[Any, int],
    direction_tag: str = ""
) -> Tuple:
    """
    Stable edge key. Ignores original MultiDiGraph keys and any insertion-dependent state.
    """
    return (
        node_rank.get(u, 10**18),
        node_rank.get(v, 10**18),
        int(data.get("flow", -1)),
        int(data.get("position", -1)),
        direction_tag,
        str(u),
        str(v),
    )


def det_sha_label(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -----------------------------
# Canonical relabeling (node IDs)
# -----------------------------

def relabel_nodes_canonically(G: nx.MultiDiGraph, rounds: int = 3) -> nx.MultiDiGraph:
    """
    Relabel nodes to 0..N-1 using only graph structure + stable attributes.

    Strategy:
      - Initial labels from stable node attributes (function/block/type/text/full_text).
      - WL-style refinement using sorted in/out neighbor labels + edge attrs for a few rounds.
      - Sort by refined label + degrees, then assign new integer ids.

    Note:
      In extremely symmetric subgraphs, multiple nodes may remain indistinguishable.
      In that case, any deterministic order depends on some external tie-breaker.
      This implementation uses the current node's string form ONLY as the last-resort
      to break exact ties among fully indistinguishable nodes.
    """
    nodes = list(G.nodes(data=True))
    labels = {n: det_sha_label(det_node_sort_key(n, d)) for n, d in nodes}

    for _ in range(max(0, rounds)):
        new_labels = {}
        for n in G.nodes():
            out_sig = []
            for _, v, k, ed in G.out_edges(n, keys=True, data=True):
                out_sig.append((
                    "o",
                    labels.get(v, ""),
                    int(ed.get("flow", -1)),
                    int(ed.get("position", -1)),
                ))
            in_sig = []
            for u, _, k, ed in G.in_edges(n, keys=True, data=True):
                in_sig.append((
                    "i",
                    labels.get(u, ""),
                    int(ed.get("flow", -1)),
                    int(ed.get("position", -1)),
                ))
            out_sig.sort()
            in_sig.sort()
            new_labels[n] = det_sha_label({
                "self": labels.get(n, ""),
                "out": out_sig,
                "in": in_sig,
            })
        labels = new_labels

    # Final ordering key
    def final_key(n: Any) -> Tuple:
        d = G.nodes[n]
        return (
            labels.get(n, ""),
            int(G.in_degree(n)),
            int(G.out_degree(n)),
            det_node_sort_key(n, d),
            str(n),  # last-resort tie-breaker only
        )

    ordered = sorted(G.nodes(), key=final_key)
    mapping = {old: new for new, old in enumerate(ordered)}
    H = nx.relabel_nodes(G, mapping, copy=True)
    return H


# -----------------------------
# Deterministic rebuild (edges + keys + ids)
# -----------------------------

def canonicalize_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Rebuild graph with deterministic node insertion and deterministic edge insertion.
    Also assigns deterministic edge keys and edge 'id' attributes (0..E-1).
    """
    H = nx.MultiDiGraph()

    # Nodes inserted deterministically
    nodes_sorted = sorted(G.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1]))
    for n, d in nodes_sorted:
        H.add_node(n, **deepcopy(d))

    node_rank = {n: i for i, (n, _) in enumerate(nodes_sorted)}

    # Collect edges ignoring existing keys; sort deterministically
    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        edges.append((u, v, deepcopy(d)))

    edges_sorted = sorted(edges, key=lambda e: det_edge_sort_key(e[0], e[1], e[2], node_rank))
    for eid, (u, v, d) in enumerate(edges_sorted):
        dd = deepcopy(d)
        dd["id"] = eid
        # IMPORTANT: explicit MultiDiGraph key avoids insertion-history key drift :contentReference[oaicite:5]{index=5}
        H.add_edge(u, v, key=eid, **dd)

    return H


def write_gexf_deterministic(G: nx.MultiDiGraph, path: str) -> None:
    """
    Write GEXF with reduced formatting variability (prettyprint=False). :contentReference[oaicite:6]{index=6}
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_gexf(G, path, prettyprint=False)


def add_nodes_and_edges_with_explicit_keys(
    G: nx.MultiDiGraph,
    new_nodes: List[Tuple[Any, Dict[str, Any]]],
    new_edges_triplets: List[Tuple[Any, Any, Dict[str, Any]]],
) -> None:
    """
    Add nodes and edges deterministically:
      - nodes added as provided (caller must use deterministic ids)
      - edges are sorted and inserted with explicit keys + id attributes
    """
    if new_nodes:
        for n, d in new_nodes:
            G.add_node(n, **deepcopy(d))

    if not new_edges_triplets:
        return

    # Build node ranking for stable edge sorting
    nodes_sorted = sorted(G.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1]))
    node_rank = {n: i for i, (n, _) in enumerate(nodes_sorted)}

    # Determine starting edge id/key
    existing_ids = []
    for _, _, _, d in G.edges(keys=True, data=True):
        if "id" in d:
            try:
                existing_ids.append(int(d["id"]))
            except Exception:
                pass
    start = (max(existing_ids) + 1) if existing_ids else G.number_of_edges()

    edges_sorted = sorted(
        [(u, v, deepcopy(d)) for (u, v, d) in new_edges_triplets],
        key=lambda e: det_edge_sort_key(e[0], e[1], e[2], node_rank),
    )

    for offset, (u, v, d) in enumerate(edges_sorted):
        eid = start + offset
        d["id"] = eid
        G.add_edge(u, v, key=eid, **d)



def process_graph(name: str, g: nx.MultiDiGraph, csv_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Deterministic replacement for process_graph:
      - Normalize node attrs (features->full_text if needed)
      - Canonical relabel (optional but recommended)
      - Canonicalize insertion order
      - Assign deterministic edge keys/ids
      - Write with prettyprint=False
    """
    g2 = nx.MultiDiGraph()

    # Normalize node attributes deterministically
    for node, ndata in sorted(g.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1])):
        attrs = deepcopy(ndata)
        if "features" in attrs and isinstance(attrs["features"], dict):
            ft = attrs["features"].get("full_text")
            if isinstance(ft, list) and ft:
                attrs["full_text"] = str(ft[0])
            # Keep original features too if you want; your downstream uses full_text.
            # attrs.pop("features", None)
        g2.add_node(node, **attrs)

    # Copy edges (keys irrelevant; will be re-keyed)
    for u, v, k, d in g.edges(keys=True, data=True):
        g2.add_edge(u, v, key=k, **deepcopy(d))

    # Prune redundant nodes using your existing logic
    prune_redundant_nodes(g2)

    # Canonical relabel (0..N-1) to stabilize node IDs across runs
    g2 = relabel_nodes_canonically(g2, rounds=3)

    # Canonicalize + assign deterministic edge keys and edge ids
    g2 = canonicalize_graph(g2)

    # Write result
    original_gexf_folder = os.path.join(processed_gexf_folder, "original")
    create_dir_if_not_exists(original_gexf_folder)
    new_gexf_file = os.path.join(original_gexf_folder, f"{name}_processed_result.gexf")
    write_gexf_deterministic(g2, new_gexf_file)

    if csv_dict is not None:
        csv_dict[name] = {
            "num_node": len(g2.nodes),
            "num_edge": len(g2.edges),
            "name": name,
        }


def graph_generator(
    name: str,
    path: str,
    benchmark: str,
    src_ext: str = "cpp",
    generate_programl: bool = False,
    csv_dict: Optional[Dict[str, Any]] = None,
    top_func: Optional[str] = None,
) -> None:
    """
    Deterministic replacement for graph_generator.
    Logic mirrors yours:
      - optionally run clang_script.sh
      - load ProGraML graph
      - parse icmp + pragmas
      - add pragma nodes/edges
      - process + write
    """
    # 1) generate ProGraML graph (same as your logic)
    if generate_programl:
        cmd = ["/bin/bash", f"{get_root_path()}/src/clang_script.sh", str(name), str(path), str(type_graph)]
        p = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
        out, err = p.communicate()
        print("returncode:", p.returncode)
        print("stdout:\n", out)
        print("stderr:\n", err)

    # 2) convert to networkx
    g_nx = llvm_to_nx(os.path.join(path, name))

    # Normalize + canonical relabel early to stabilize downstream matching/order
    g_nx = canonicalize_graph(g_nx)
    g_nx = relabel_nodes_canonically(g_nx, rounds=3)
    g_nx = canonicalize_graph(g_nx)

    g_nx_nodes, g_nx_edges = g_nx.number_of_nodes(), g_nx.number_of_edges()
    print(f"number of nodes: {g_nx_nodes} and number of edges: {g_nx_edges}")

    # Optional: write raw canonicalized graph (deterministic)
    graph_path = os.path.join(path, f"{name}.gexf")
    write_gexf_deterministic(g_nx, graph_path)

    # 3) loop detection in llvm + pragmas in source
    for_dict_llvm, for_count_llvm = get_icmp(path, name)
    for_dict_source, for_count_source = get_pragmas_loops(
        path,
        f"{name}_placeholders",
        EXT=src_ext,
    )
    assert for_count_llvm == for_count_source, (
        f"the number of for loops from the LLVM code and source code do not match "
        f"{for_count_llvm} in llvm vs {for_count_source} in the code"
    )

    # 4) add pragma nodes/edges deterministically
    kernel_info_file = os.path.join(path, "kernel_info.txt")
    new_nodes, new_edges = create_pragma_nodes(
        g_nx, g_nx.number_of_nodes(), kernel_info_file, for_dict_source, for_dict_llvm
    )

    # create_pragma_nodes returns:
    #   new_nodes: [(nid, dict), ...]
    #   new_edges: [(u, v, attrdict), ...]  <-- no explicit keys
    add_nodes_and_edges_with_explicit_keys(g_nx, new_nodes, new_edges)

    # 5) process/write deterministically
    process_graph(name, g_nx, csv_dict=csv_dict)


def add_auxiliary_nodes(
    name: str,
    path: str,
    processed_path: str,
    csv_dict: Optional[Dict[str, Any]],
    node_type: str = "block",
    connected: bool = False,
) -> Optional[None]:
    """
    Deterministic replacement for add_auxiliary_nodes with explicit edge keys/ids.
    Mirrors your logic but:
      - rebuilds edges with explicit keys/ids
      - writes with prettyprint=False
    """
    if node_type != "block":
        raise NotImplementedError()

    gexf_file = os.path.join(path, "original", f"{name}_processed_result.gexf")
    new_gexf_file = os.path.join(processed_path, f"{name}_processed_result.gexf")
    if not os.path.isfile(gexf_file):
        print(f'Processed graph not found for kernel "{name}": {gexf_file} — skipping')
        return None

    print(f"processing {gexf_file}")
    g = nx.readwrite.gexf.read_gexf(gexf_file, node_type=str)

    # Normalize read graph into MultiDiGraph and canonicalize
    g0 = nx.MultiDiGraph()
    for n, d in g.nodes(data=True):
        g0.add_node(n, **deepcopy(d))
    for u, v, d in g.edges(data=True):
        g0.add_edge(u, v, **deepcopy(d))
    g0 = canonicalize_graph(g0)
    g0 = relabel_nodes_canonically(g0, rounds=3)
    g0 = canonicalize_graph(g0)

    prev_nodes, prev_edges = g0.number_of_nodes(), g0.number_of_edges()
    # print(f"started with {prev_nodes} nodes and {prev_edges} edges")

    current_g_value = {
        "name": name,
        "prev_node": prev_nodes,
        "prev_edge": prev_edges,
    }

    g_new = nx.MultiDiGraph()
    # Copy nodes
    for n, d in g0.nodes(data=True):
        g_new.add_node(n, **deepcopy(d))

    # Copy edges deterministically (will re-key)
    for u, v, k, d in g0.edges(keys=True, data=True):
        g_new.add_edge(u, v, **deepcopy(d))

    # Deterministic edge id start
    next_eid = g_new.number_of_edges()

    block_nodes: Dict[str, Dict[str, Any]] = {}
    block_func: Dict[Any, Dict[str, Any]] = {}
    next_node_id = g_new.number_of_nodes()

    # Deterministic traversal
    for node, ndata in sorted(g_new.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1])):
        key = f"function-{ndata['function']}-block-{ndata['block']}"
        if key not in block_nodes:
            new_node_obj = create_pseudo_node_block(ndata["block"], ndata["function"])
            block_nodes[key] = {"id": next_node_id, "node": new_node_obj, "last_position": 0}
            g_new.add_node(next_node_id, **new_node_obj.get_attr(after_process=True))
            next_node_id += 1

        # Track block counts
        f = ndata["function"]
        b = ndata["block"]
        if f not in block_func:
            block_func[f] = {"count": 1, "blocks": [b]}
        elif b not in block_func[f]["blocks"]:
            block_func[f]["count"] += 1
            block_func[f]["blocks"].append(b)

        pseudo_id = block_nodes[key]["id"]
        pos = block_nodes[key]["last_position"]

        # Add bidirectional edges with explicit keys/ids
        e1 = {"id": next_eid, "flow": 4, "position": pos}
        g_new.add_edge(node, pseudo_id, key=next_eid, **e1)
        next_eid += 1
        e2 = {"id": next_eid, "flow": 4, "position": pos}
        g_new.add_edge(pseudo_id, node, key=next_eid, **e2)
        next_eid += 1

        block_nodes[key]["last_position"] = pos + 1

    if connected:
        # Connect pseudo nodes in a deterministic order
        sorted_keys = sorted(block_nodes.keys(), key=natural_keys)
        for a, b in zip(sorted_keys[:-1], sorted_keys[1:]):
            id1 = block_nodes[a]["id"]
            id2 = block_nodes[b]["id"]
            e1 = {"id": next_eid, "flow": 5, "position": 0}
            g_new.add_edge(id1, id2, key=next_eid, **e1)
            next_eid += 1
            e2 = {"id": next_eid, "flow": 5, "position": 0}
            g_new.add_edge(id2, id1, key=next_eid, **e2)
            next_eid += 1

    # Prune + canonicalize + write
    prune_redundant_nodes(g_new)
    g_new = relabel_nodes_canonically(g_new, rounds=3)
    g_new = canonicalize_graph(g_new)

    new_nodes_ct, new_edges_ct = g_new.number_of_nodes(), g_new.number_of_edges()
    # print(f"ending with {new_nodes_ct} nodes and {new_edges_ct} edges")

    current_g_value["new_node"] = new_nodes_ct
    current_g_value["new_edge"] = new_edges_ct
    current_g_value["block"] = sum(v["count"] for v in block_func.values())

    if csv_dict is not None:
        csv_dict[name] = current_g_value

    write_gexf_deterministic(g_new, new_gexf_file)
    return None


def augment_graph_hierarchy(
    name: str,
    for_blocks_info: Dict[str, Any],
    src_path: str,
    dst_path: str,
    csv_dict: Optional[Dict[str, Any]] = None,
    node_type: str = "block",
) -> None:
    """
    Deterministic replacement for augment_graph_hierarchy:
      - reads gexf
      - adds hierarchy edges with explicit keys/ids
      - canonicalizes and writes deterministically
    """
    src_dir = os.path.join(get_root_path(), f'{type_graph}/{name}')
    llvm_func_bodies = _parse_llvm_function_bodies(os.path.join(src_dir, 'kernel_info.txt'))

    if node_type != "block":
        raise NotImplementedError()

    gexf_file = os.path.join(src_path, f"{name}_processed_result.gexf")
    new_gexf_file = os.path.join(dst_path, f"{name}_processed_result.gexf")
    # print(f"processing {gexf_file}")

    g = nx.readwrite.gexf.read_gexf(gexf_file, node_type=str)
    g0 = nx.MultiDiGraph()
    for n, d in g.nodes(data=True):
        g0.add_node(n, **deepcopy(d))
    for u, v, d in g.edges(data=True):
        g0.add_edge(u, v, **deepcopy(d))
    g0 = canonicalize_graph(g0)
    g0 = relabel_nodes_canonically(g0, rounds=3)
    g0 = canonicalize_graph(g0)

    prev_nodes, prev_edges = g0.number_of_nodes(), g0.number_of_edges()
    # print(f"started with {prev_nodes} nodes and {prev_edges} edges")

    current_g_value = {
        "name": name,
        "prev_node": prev_nodes,
        "prev_edge": prev_edges,
    }

    g_new = canonicalize_graph(g0)
    next_eid = g_new.number_of_edges()

    # Determine block ids deterministically (reuse your logic but deterministic traversal)
    block_ids: Dict[str, Tuple[Any, Any]] = {}
    for for_l in sorted(for_blocks_info.keys()):
        info = for_blocks_info[for_l]
        expected_graph_func_id = infer_graph_function_id(g_new, info.get("llvm_func"), llvm_func_bodies)
        if expected_graph_func_id is None:
            raise RuntimeError(
                f"Could not infer graph function id for hierarchy loop '{for_l}' "
                f"(LLVM function: {info.get('llvm_func')}). Refusing global fallback."
            )
        matches = []

        for node, ndata in sorted(g_new.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1])):
            if expected_graph_func_id is not None and int(ndata.get("function", -1)) != int(expected_graph_func_id):
                continue

            ft = det_get_full_text(ndata)
            if not ft:
                continue

            if info["next_instr"][0] in ft:
                correct = 1
                for nb in sorted(g_new.neighbors(node), key=lambda n: det_node_sort_key(n, g_new.nodes[n])):
                    if info["next_instr"][1] in det_get_full_text(g_new.nodes[nb]):
                        correct += 1
                        if correct == 2:
                            for nb2 in sorted(g_new.neighbors(nb), key=lambda n: det_node_sort_key(n, g_new.nodes[n])):
                                if info["next_instr"][2] in det_get_full_text(g_new.nodes[nb2]):
                                    correct += 1
                                    break
                    if correct == 3:
                        break

                if correct == 3:
                    matches.append((node, ndata))

        if len(matches) == 0:
            raise RuntimeError(f"could not find the respective block for label {for_l}")

        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous hierarchy match for {for_l} in function {info.get('llvm_func')}: "
                f"{[(m[0], m[1].get('block'), m[1].get('function')) for m in matches]}"
            )

        node, ndata = matches[0]
        block_ids[for_l] = (ndata["block"], ndata["function"])


    # Find pseudo node for each loop block deterministically
    node_ids_block: Dict[str, Any] = {}
    for for_l in sorted(for_blocks_info.keys()):
        b, f = block_ids[for_l]
        for node, ndata in sorted(g_new.nodes(data=True), key=lambda nd: det_node_sort_key(nd[0], nd[1])):
            if "pseudo_block" in str(ndata.get("text", "")) and ndata.get("block") == b and ndata.get("function") == f:
                node_ids_block[for_l] = node
                break

    # Add hierarchy edges with explicit keys/ids
    for for_l in sorted(for_blocks_info.keys()):
        children = for_blocks_info[for_l].get("children", [])
        if not children:
            continue
        id1 = node_ids_block[for_l]
        position = 0
        for child in children:
            id2 = node_ids_block[child]
            e1 = {"id": next_eid, "flow": 6, "position": position}
            g_new.add_edge(id1, id2, key=next_eid, **e1)
            next_eid += 1
            e2 = {"id": next_eid, "flow": 6, "position": position}
            g_new.add_edge(id2, id1, key=next_eid, **e2)
            next_eid += 1
            position += 1

    prune_redundant_nodes(g_new)
    g_new = relabel_nodes_canonically(g_new, rounds=3)
    g_new = canonicalize_graph(g_new)

    new_nodes_ct, new_edges_ct = g_new.number_of_nodes(), g_new.number_of_edges()
    # print(f"ending with {new_nodes_ct} nodes and {new_edges_ct} edges")

    current_g_value["new_node"] = new_nodes_ct
    current_g_value["new_edge"] = new_edges_ct
    if csv_dict is not None:
        csv_dict[name] = current_g_value

    write_gexf_deterministic(g_new, new_gexf_file)


def run_graph_gen(mode: str = "initial", connected: bool = True) -> None:
    """
    Deterministic run_graph_gen for each mode : initial/auxiliary/hierarchy.
    """
    # Keep your paths/logic as in run_graph_gen
    test = "original"
    global_processed_gexf_folder = processed_gexf_folder

    base_dataset_dir = "/home/ubuntu/Data4LLMPrompting/ApplicationDataset"
    csvs_dir = "/home/ubuntu/Data4LLMPrompting/preprocessed_CSVS"
    source_map_csv = "/home/ubuntu/Data4LLMPrompting/ApplicationInformation.csv"

    if mode == "initial":
        source_map = load_kernel_source_map(source_map_csv)

    if mode == "initial":
        csv_header = ["name", "num_node", "num_edge"]
    else:
        csv_header = ["name", "prev_node", "prev_edge", "new_node", "new_edge"]
    if mode == "auxiliary":
        csv_header.append("block")
    csv_dict = {"header": csv_header}

    if mode == "initial":
        for kernel in sorted(os.listdir(base_dataset_dir)):
            kernel_path = os.path.join(base_dataset_dir, kernel)

            if kernel not in source_map:
                raise RuntimeError(f"No source mapping found in CSV for kernel '{kernel}'")
            info = source_map[kernel]
            orig_src_name = info["file_name"]
            ext = info["ext"]
            src_ext = ext.lstrip(".")
            top_func = info["top"]

            src_path = os.path.join(kernel_path, orig_src_name)
            if not os.path.isfile(src_path):
                raise FileNotFoundError(f"Mapped source file '{src_path}' does not exist for kernel '{kernel}'")

            header_files = list(iglob(os.path.join(kernel_path, "*.h"), recursive=False))
            # stable selection
            kernel_info_candidates = sorted(glob.glob(os.path.join(kernel_path, "kernel_info.txt")))
            if not kernel_info_candidates:
                raise FileNotFoundError(f"No kernel_info.txt found in {kernel_path}")
            kernel_info_file = kernel_info_candidates[0]

            print("####################")
            print("Now processing", kernel)
            harp_kernel_dir = os.path.join(get_root_path(), f"{type_graph}/{kernel}")

            if not os.path.exists(harp_kernel_dir):
                create_dir_if_not_exists(harp_kernel_dir)

                new_src_path = os.path.join(harp_kernel_dir, f"{kernel}{ext}")
                shutil.copyfile(src_path, new_src_path)

                for header_file in sorted(header_files):
                    new_header_path = os.path.join(harp_kernel_dir, os.path.basename(header_file))
                    shutil.copyfile(header_file, new_header_path)

                new_kernel_info_path = os.path.join(harp_kernel_dir, "kernel_info.txt")
                shutil.copyfile(kernel_info_file, new_kernel_info_path)

                placeholder_lines = insert_placeholders(new_src_path)
                placeholders_src_path = os.path.join(harp_kernel_dir, f"{kernel}_placeholders{ext}")
                with open(placeholders_src_path, "w") as f:
                    f.writelines(placeholder_lines)

            if not os.path.exists(global_processed_gexf_folder):
                create_dir_if_not_exists(global_processed_gexf_folder)

            graph_generator(
                kernel,
                harp_kernel_dir,
                kernel,
                src_ext=src_ext,
                generate_programl=True,
                csv_dict=csv_dict,
                top_func=top_func,
            )
            write_csv_file(csv_dict, csv_header, f"{type_graph}/{mode}.csv")

    elif mode == "auxiliary":
        if connected:
            auxiliary_node_gexf_folder = os.path.join(
                get_root_path(),
                f"{type_graph}/processed/extended-pseudo-block-connected/",
            )
        else:
            auxiliary_node_gexf_folder = os.path.join(
                get_root_path(),
                f"{type_graph}/processed/extended-pseudo-block-base/",
            )
        create_dir_if_not_exists(auxiliary_node_gexf_folder)

        for kernel in sorted(os.listdir(base_dataset_dir)):
            print("####################")
            print("Now processing", kernel)
            add_auxiliary_nodes(
                kernel,
                global_processed_gexf_folder,
                auxiliary_node_gexf_folder,
                csv_dict=csv_dict,
                node_type="block",
                connected=connected,
            )
            print()

        write_csv_file(csv_dict, csv_header, f"{type_graph}/{mode}_{connected}.csv")

    elif mode == "hierarchy":
        auxiliary_node_gexf_folder = os.path.join(
            get_root_path(),
            f"{type_graph}/processed/extended-pseudo-block-connected/",
        )
        dest_path = os.path.join(
            get_root_path(),
            f"{type_graph}/processed/extended-pseudo-block-connected-hierarchy/",
        )
        create_dir_if_not_exists(dest_path)
        assert os.path.exists(auxiliary_node_gexf_folder)

        for kernel in sorted(os.listdir(base_dataset_dir)):
            llvm_kernel_dir = os.path.join(get_root_path(), f"{type_graph}/{kernel}")
            print("####################")
            print("now processing", kernel)
            for_blocks_info = get_for_blocks_info(kernel, llvm_kernel_dir)
            augment_graph_hierarchy(
                kernel,
                for_blocks_info,
                src_path=auxiliary_node_gexf_folder,
                dst_path=dest_path,
                csv_dict=csv_dict,
            )
            print()

        write_csv_file(csv_dict, csv_header, f"{type_graph}/{mode}.csv")

    else:
        raise NotImplementedError()



if __name__ == "__main__":
    run_graph_gen(mode="initial", connected=True)
    run_graph_gen(mode="auxiliary", connected=False)
    run_graph_gen(mode="auxiliary", connected=True)
    run_graph_gen(mode="hierarchy", connected=True)

