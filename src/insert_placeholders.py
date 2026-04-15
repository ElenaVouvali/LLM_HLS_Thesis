import pandas as pd
import re


def insert_placeholders(cpp_file):
    with open(cpp_file, 'r') as f:
        code_lines = f.readlines()

    output = []
    i = 0
    while i < len(code_lines):
        line = code_lines[i]
        stripped = line.rstrip('\n')

        # /*L1:*/   or   L1:
        m = re.match(r'^\s*(?:/\*\s*(L\d+)\s*:\s*\*/|(L\d+)\s*:)', stripped)
        if not m:
            output.append(line)
            i += 1
            continue

        label = m.group(1) or m.group(2)
        after = stripped[m.end():]

        output.append(line)

        if re.search(r'\bfor\s*\(', after):
            output.append(f'#pragma HLS pipeline II=auto{{_PIPE_{label}}}\n')
            output.append(f'#pragma HLS unroll factor=auto{{_UNROLL_{label}}}\n')
            i += 1
            continue

        arr_match = re.search(
            r'^(?:\s*[A-Za-z_]\w*\s*:\s*)?'          # optional local label "Xyz:"
            r'\s*([A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)*)' # type tokens (e.g., "float", "const int")
            r'\s+([A-Za-z_]\w*)'                     # var name (e.g., "sin_angle")
            r'\s*(\[[^\]]+\])+'                      # one or more [..] (supports int a[2][16])
            r'(?:\s*=\s*[^;]+)?'                     # OPTIONAL initializer "= {...}" or similar
            r'\s*;'                                  # terminating ';'
        , after)

        if arr_match:
            varname = arr_match.group(2)
            output.append(
                f'#pragma HLS array_partition variable={varname} '
                f'type=auto{{_ARRAY_T_{label}}} factor=auto{{_ARRAY_F_{label}}} dim=auto{{_ARRAY_D_{label}}}\n'
            )
            i += 1
            continue

        i += 1

    return output







