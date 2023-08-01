from string import ascii_uppercase as auc
import json


def filter_excel_data(data):
    filtered_data = []
    for i in range(len(data)):
        d = data[i]

        # split heads
        parts = list(map(lambda x: x.strip(), data[i]['head'].split('|')))

        # filter out spaces in each header beside ":"
        no_column_name_parts = []
        for j in range(len(parts)):
            if ":" in parts[j]:
                parts[j] = ":".join(list(map(lambda x: x.strip(), parts[j].split(':'))))
            elif "：" in parts[j]:
                parts[j] = ":".join(list(map(lambda x: x.strip(), parts[j].split('：'))))
            else:
                no_column_name_parts.append(j)

        if no_column_name_parts:
            if no_column_name_parts == list(range(len(parts) - 2)):
                for j, a in zip(range(len(parts) - 2), auc):
                    parts[j] = f"{a}:{parts[j]}"
            else:
                is_garbage = True
                print(f"non consistent headers: {parts}")
                continue

        # wrong header, no ROWS information, throw this data
        if not any([p.upper().startswith('ROWS') for p in parts]):
            print(f"no row information: {parts}")
            continue
        # filter out texts after ROWS information
        while not parts[-1].upper().startswith('ROWS'):
            parts.pop(-1)
        parts[-1] = parts[-1].upper()
        if not parts[-1][-1].isdigit():
            parts[-1] = parts[-1].split(',')[0]
        if not parts[-1][-1].isdigit():
            parts[-1] = parts[-1].split(' ')[0]
        parts[-1] = parts[-1].replace(" ", "")

        if not parts[-1][-1].isdigit():
            print(f"non digit row specifier: {parts}")
            continue

        if not any([p.upper().startswith("HEAD") for p in parts]):
            parts.insert(-1, "HEAD:1")
        elif any([p.upper().startswith("HEADER") for p in parts]):
            idx = [p.upper().startswith("HEADER") for p in parts].index(True)
            parts[idx] = parts[idx].replace("HEADER", "HEAD")
        else:
            parts[-2] = parts[-2].upper()

        n_head_rows = parts[-2].split("HEAD:")[-1]
        if not n_head_rows.isdigit():
            print(f"unkown head specification: {n_head_rows} {parts}")
            continue

        data_rows = parts[-1].split("ROWS:")[-1]
        if '-' in data_rows:
            data_start, data_end = data_rows.split('-')
            data_start = str(max(int(data_start), int(n_head_rows) + 1))
            assert data_start.isdigit(), data_start
            assert data_end.isdigit(), data_end
        else:
            data_rows = data_rows.replace(',', '')
            assert data_rows.isdigit(), (parts, d['code'])
            data_start = str(int(n_head_rows) + 1)
            data_end = data_rows

        # print(">>>>", parts)
        parts.pop(-1)
        parts.pop(-1)

        is_garbage = False

        new_head = ""
        head_names = set()
        for p, a in zip(parts, auc):
            if not p.startswith(f"{a}:"):
                is_garbage = True
                print(f"Unknown head specifier: {p} {parts}")
                break
            else:
                head_name = p.split(f"{a}:")[1].strip()
                if head_name in head_names:
                    is_garbage = True
                    print(f"duplicate head name: {head_name} {head_names}")
                    break
                else:
                    head_names.add(head_name)
                new_head += f"{a}:{head_name}|"

        min_n_headers = 1
        if not is_garbage and len(head_names) <= min_n_headers:
            print(f"too few head names: {parts}")
        is_garbage |= len(head_names) <= min_n_headers

        new_head += f"HEAD:{n_head_rows}|"
        new_head += f"ROWS:{data_start}-{data_end}"
        # new_head += f"{DATA_ROW_END_TOKEN}{data_end}"

        if not is_garbage:
            # print(new_head)
            filtered_data.append(
                dict(head=new_head,
                     task=data[i]['task'],
                     **{
                         k: v
                         for k, v in data[i].items() if k != 'task' and k != 'head'
                     }))

    print(f"The original number of data entries: {len(data)}, number after filtering: {len(filtered_data)}")
    return filtered_data


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, nargs='+', required=True)
parser.add_argument("--output", type=str, nargs="+", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    for input_fn, output_fn in zip(args.input, args.output):
        with open(input_fn, 'r') as f:
            original_data = json.load(f)
        filtered_data = filter_excel_data(original_data)
        with open(output_fn, 'w') as f:
            json.dump(filtered_data, f)
            print(f"save file at {output_fn}")
