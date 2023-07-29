import os
import re
import subprocess

exp_name = "wps-rw-s1"
cfgs = []
for trial in range(1, 55):
    log_root = f"/home/aigc/llm/fw/logs/{exp_name}_lora-sweep-20230728-{trial}/"
    s = subprocess.check_output(f"cat {os.path.join(log_root, 'model_worker')} | grep 'Configuring with'",
                                shell=True).decode('ascii').strip().split('\n')[-1].strip()

    cfg = dict()
    for k in ['lr', 'weight_decay', 'lora_scaling', 'lora_dim']:
        match_e = re.search(rf"'{k}': (\d+\.\d+e-\d+)", s)
        match_f = re.search(rf"'{k}': (\d+\.\d+)", s)
        match_int = re.search(rf"'{k}': (\d+)", s)
        if match_e is not None:
            v = match_e.group(1)
        elif match_f is not None:
            v = match_f.group(1)
        else:
            v = match_int.group(1)
        cfg[k] = float(v)

    eval_s = subprocess.check_output(f"cat {os.path.join(log_root, 'master_worker')} | grep 'Evaluation'",
                                     shell=True).decode('ascii').strip().split('\n')[-1].strip()
    match_f = re.search(r"'acc': (\d+\.\d+)", eval_s)
    cfg['acc'] = float(match_f.group(1))
    cfg['lr_scaling_product'] = float(cfg['lr']) * float(cfg['lora_scaling'])

    cfgs.append(cfg)

cfgs = [{k: float(v) for k, v in x.items()} for x in cfgs]


def pretty_print(data):
    # get all keys
    keys = data[0].keys()
    # calculate max width for each column
    widths = {}
    for k in keys:
        widths[k] = max(len(str(d[k])) for d in data) + 2
    # print header row
    print("| " + " | ".join("{:<{width}}".format(k, width=widths[k]) for k in keys) + " |")
    # print separator row
    print("-" * (sum(widths.values()) + 3 * (len(keys) - 1)))
    # print data rows
    for d in data:
        print("| " + " | ".join("{:<{width}}".format(d[k], width=widths[k]) for k in keys) + " |")


# lora dim=8, lora_scaling
pretty_print(sorted([cfg for cfg in cfgs if cfg['lora_dim'] == 128], key=lambda x: x['acc']))
# pretty_print(sorted([cfg for cfg in cfgs if cfg['lr'] <= 1e-3], key=lambda x: x['acc']))
