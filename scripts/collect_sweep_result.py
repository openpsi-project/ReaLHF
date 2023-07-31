import os
import re
import subprocess

exp_name = "wps-rw-s1"
trial_name_prefix = "lora-sweep-20230731"
log_dir_path = "/home/aigc/llm/fw/logs/"

count = 0
for root, dirs, files in os.walk(log_dir_path):
    for name in dirs:
        if name.startswith(f"{exp_name}_{trial_name_prefix}"):
            count += 1

keys = ['betas', 'lr_scheduler_type', 'min_lr_ratio', 'warmup_steps_proportion']
dtypes = [str, str, float, float]
formatter = {
    float: "{:.4f}",
    int: "{:d}",
    str: "{}",
}

cfgs = []
for trial in range(1, count):
    log_root = os.path.join(log_dir_path, f"{exp_name}_{trial_name_prefix}-{trial}")
    s = subprocess.check_output(f"cat {os.path.join(log_root, 'model_worker')} | grep 'Configuring with'",
                                shell=True).decode('ascii').strip().split('\n')[-1].strip()

    cfg = dict()
    for k, dtype in zip(keys, dtypes):
        match_e = re.search(rf"'{k}': (\d+\.\d+e-\d+)", s)
        match_f = re.search(rf"'{k}': (\d+\.\d+)", s)
        match_int = re.search(rf"'{k}': (\d+)", s)
        match_s = re.search(rf"'{k}': ([^)]+)", s)
        if dtype == float:
            if match_e is not None:
                v = match_e.group(1)
            else:
                v = match_f.group(1)
        elif dtype == int:
            v = match_int.group(1)
        elif dtype == str:
            v = match_s.group(1)
        else:
            raise NotImplementedError()
        cfg[k] = dtype(v)

    eval_s = subprocess.check_output(f"cat {os.path.join(log_root, 'master_worker')} | grep 'Evaluation'",
                                     shell=True).decode('ascii').strip().split('\n')[-1].strip()
    match_f = re.search(r"'acc': (\d+\.\d+)", eval_s)
    cfg['acc'] = float(match_f.group(1))

    master_cfg_s = subprocess.check_output(
        f"cat {os.path.join(log_root, 'master_worker')} | grep 'Configuring with'",
        shell=True).decode('ascii').strip().split('\n')[-1].strip()
    match_s = re.search(rf"total_train_epochs=(\d+)", master_cfg_s)
    cfg['total_train_epochs'] = match_s.group(1)

    cfgs.append(cfg)


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
pretty_print(sorted([cfg for cfg in cfgs], key=lambda x: x['acc']))
# pretty_print(sorted([cfg for cfg in cfgs if cfg['lr'] <= 1e-3], key=lambda x: x['acc']))
