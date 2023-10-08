import os
from typing import List, Dict, Any
from collections import defaultdict
import json
import matplotlib.pyplot as plt

LLM_LOGS = os.environ["LLM_LOGS"]
DSCHAT_LOGS = "/data/aigc/llm/logs/dschat/meizy"
CONFIG_KEYS_TO_PRINT = [
    "actor_model_name", "critic_model_name", "actor_zero_stage", "critic_zero_stage",
    "hybrid_engine", "offload_critic_param", "offload_critic_optimizer_states", "offload_ref",
    "offload_reward"
]
LOGFILE_NAME_TO_CHECK = ["master_worker", "model_worker", "rlhf"]
DATA_NAME_TO_CHECK = ["e2e", "level1", "level2", "level3"]

spec_to_n_params = {
    (5120, 60): "19b",
    (6144, 40): "18.4b",
    (6144, 50): "23b",
    (6144, 60): "27.5b",
    (9216, 64): "66b",
    (7168, 48): "30b",
    (5120, 40): "13b",
    (5120, 50): "16b",
    (4096, 32): "6.7b",
    (2560, 32): "2.7b",
    (2048, 24): "1.3b",
    (1024, 24): "350m",
    (768, 12): "125m",
}

def dirname_to_n_params(name: str):
    x, y = name.split("-")[1:]
    return spec_to_n_params[(int(x), int(y))]

def parse_data_line(line: str) -> Dict[str, Any]:
    words = line.split(" ")
    data = dict()
    # print(line)
    k = v1 = v2 = None
    for w in words:
        # if "oom" in w or "OOM" in w:
        #     k = "e2e"
        #     v1 = "oom"
        #     break
        w = w.replace("\n", "").strip()    
        if w.count("#") == 2:
            k = w.split("#")[1]
        if w.count("*") == 2:
            v1 = w.split("*")[1]
        if w.count("$") == 2:
            v2 = w.split("$")[1]
    if k == None or v1 == None:
        return {} 
    # print(k, v1, v2)
    v = (v1, v2) if v2 is not None else v1
    data[k] = v
    return data

def parse_data_file(filename: str) -> Dict[str, List]:
    try:
        fp = open(filename, "r")
    except FileNotFoundError:
        return {}
    data = defaultdict(list)
    lines = fp.readlines()
    for line in lines:
        data_line = parse_data_line(line)
        # filter
        for k, v in data_line.items():
            if isinstance(v, tuple):
                if not v[0].replace('.', '', 1).isdigit():
                    continue
            data[k].append(v)
    return data

def parse_config(directory: str, config_filename: str = "config.json") -> Dict[str, Any]:
    try:
        fp = open(os.path.join(directory, config_filename), "r")
    except FileNotFoundError:
        return {}
    config = json.load(fp)
    return config

def print_config(config: Dict[str, Any]):
    # print(config)
    if len(config) == 0:
        return 
    for k in CONFIG_KEYS_TO_PRINT:
        print(f"{k}: {config[k]}")

def collect_data():
    # parse_data_file("/data/aigc/llm/logs/meizy/opt-2x8-offloadref-22_20231001-00/master_worker-0")
    count = 0
    all_data = {}
    for dn in os.listdir(LLM_LOGS):
        if not dn.startswith("opt") and not dn.startswith("starcoder"):
            continue
        if not "params" in dn or not "20231008-01" in dn:
            continue
        if "TEST" in dn or "1x1" in dn:
            continue
        s = ""
        this_data = {}
        for fn in os.listdir(os.path.join(LLM_LOGS, dn)):
            if not any([fn.startswith(n) for n in LOGFILE_NAME_TO_CHECK]):
                continue
            # print(fn)
            data = parse_data_file(os.path.join(LLM_LOGS, dn, fn))
            for k, v in data.items():
                if k in DATA_NAME_TO_CHECK:
                    s += f"{k}: {v}\n"
            if len(data) > 0:
                this_data.update(data)
        if s != "":
            count += 1
            print(dn)
            config = parse_config(os.path.join(LLM_LOGS, dn))
            print_config(config)
            print(s)
        else:
            continue
        all_data[dn] = (config, this_data)
    return all_data    

def collect_dschat_data():
    count = 0
    all_data = {}
    for dn in os.listdir(DSCHAT_LOGS):
        if "20231008-02" not in dn:
            continue
        s = ""
        this_data = {}
        for fn in os.listdir(os.path.join(DSCHAT_LOGS, dn)):
            if not any([fn.startswith(n) for n in LOGFILE_NAME_TO_CHECK]):
                continue
            # print(fn)
            data = parse_data_file(os.path.join(DSCHAT_LOGS, dn, fn))
            for k, v in data.items():
                if k in DATA_NAME_TO_CHECK:
                    s += f"{k}: {v}\n"
            if len(data) > 0:
                this_data.update(data)
        if s != "":
            count += 1
            print(dn)
            config = parse_config(os.path.join(DSCHAT_LOGS, dn))
            print_config(config)
            print(s)
        else:
            continue
        all_data[dn] = (config, this_data)
    return all_data  


def plot_data(data: dict, dschat_data: dict, fn: str):
    xs = []
    xticklabels = []
    ys = []
    cmts = []
    for dn, (cfg, dp) in data.items():
        short_dn = dn.split("_")[0].split("-")[-1]
        # print(dn, dp)
        if "e2e" not in dp:
            continue
        if len(dp["e2e"]) > 0:
            xs.append(short_dn)
            ys.append(dp["e2e"][0])
            # cmts.append(f"stages: {cfg['actor_zero_stage']}-{cfg['critic_zero_stage']}; hybrid: {cfg['hybrid_engine']};" 
            #             f"offload: critic {cfg['offload_critic_param']} ref: {cfg['offload_ref']} rew: {cfg['offload_reward']}")
            xticklabels.append(f"{dirname_to_n_params(cfg['actor_model_name'])}+{dirname_to_n_params(cfg['critic_model_name'])}")

    xs = list(range(len(xs)))
    ys = [float(y) for y in ys]

    def parse_param(x):
        param1 = x.split("+")[0]
        param2 = x.split("+")[1]
        def unify_param_unit(param):
            if "b" in param:
                return float(param.strip("b")) * 1000
            if "m" in param:
                return float(param.strip("m"))
            return float(param)
        return (unify_param_unit(param1), unify_param_unit(param2))

    unsorted = list(zip(xs, ys, xticklabels))
    xsys = sorted(unsorted, key=lambda x: parse_param(x[2]))
    xs = [x for x, y, c in xsys]
    ys = [y for x, y, c in xsys] 
    xticklabels = [c for x, y, c in xsys]

    dschat_ys = [0 for y in ys]
    dschat_z3_ys = [0 for y in ys]

    for dn, (cfg, dp) in dschat_data.items():
        if "e2e" not in dp:
            continue
        if len(dp["e2e"]) > 0:
            if cfg["actor_zero_stage"] == 2:
                xticklabel = f"{dirname_to_n_params(cfg['actor_model_name'])}+{dirname_to_n_params(cfg['critic_model_name'])}"
                if xticklabel in xticklabels:
                    idx = xticklabels.index(xticklabel)
                    dschat_ys[idx] = float(dp["e2e"][0])
            elif cfg["actor_zero_stage"] == 3:
                xticklabel = f"{dirname_to_n_params(cfg['actor_model_name'])}+{dirname_to_n_params(cfg['critic_model_name'])}"
                if xticklabel in xticklabels:
                    idx = xticklabels.index(xticklabel)
                    dschat_z3_ys[idx] = float(dp["e2e"][0])

    print(xs)
    print(ys)
    print(dschat_ys)
    print(dschat_z3_ys)
    print(xticklabels)
    fig = plt.figure(figsize=(8, 4))
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.35, top=0.95)

    ys = [14/y for y in ys]
    dschat_ys = [16/y if y != 0 else 0 for y in dschat_ys]
    dschat_z3_ys = [16/y if y != 0 else 0 for y in dschat_z3_ys]

    plt.bar(list(range(len(ys))), ys, width=0.3, label="ours")
    plt.bar([x+0.3 for x in list(range(len(ys)))], dschat_ys, width=0.3, label="dschat z2")
    plt.bar([x+0.6 for x in list(range(len(ys)))], dschat_z3_ys, width=0.3, label="dschat z3")
    plt.legend()
    # plt.xlabel("13b+1.3b")
    plt.xticks([x+0.3 for x in list(range(len(ys)))], xticklabels, rotation=45)
    plt.ylabel("seqs per second")
    # strs = ["Configs: "]
    # for i, c in enumerate(cmts):
    #     strs.append(f"cfg {xs[i]}: {c};; ")
    
    # s = "\n".join(strs)
    # plt.text(15, 0, s, fontsize=11)
    
    plt.savefig(f"figs/{fn}.png")

def main():
    all_data = collect_data()
    dschat_data = collect_dschat_data()
    new_dschat_data = {}

    def parse_e2e(list_of_data):
        float_list = [float(x) for x in list_of_data]
        return sum(float_list) / len(float_list)

    for k, (cfg, dp) in dschat_data.items():
        if 'e2e' in dp:
            new_data = {'e2e': parse_e2e(dp['e2e'])}
            new_dschat_data[k] = (cfg, new_data)
    # print(new_dschat_data)
    plot_data(all_data, dschat_data, "params")

if __name__ == "__main__":
    main()