import argparse
import collections
import json
import pickle
import random
import time

import torch
import transformers
import zmq

import api.utils

parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", type=int, default=7777)
args = parser.parse_args()

if __name__ == "__main__":
    model_path = "/data/marl/checkpoints/fw/starcoder-wps-best/"
    max_answer_len = 512
    generation_kwargs = dict(max_new_tokens=max_answer_len,
                             do_sample=False,
                             top_p=1.0,
                             top_k=100,
                             temperature=1.0,
                             num_beams=1,
                             num_beam_groups=1,
                             num_return_sequences=1)
    device = torch.device("cuda:0")
    tokenizer = api.utils.load_hf_tokenizer(model_path)
    print("Loading model...", flush=True)
    model = api.utils.create_hf_nn(
        transformers.AutoModelForCausalLM,
        model_name_or_path=model_path,
        generation_kwargs=generation_kwargs,
    ).to(device)

    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    print("Ready to serve.", flush=True)
    while True:
        prompt = pickle.loads(socket.recv())
        print(f"Receive request: {prompt}", flush=True)
        token = tokenizer(prompt, return_tensors="pt").to(device)

        generate_ids = model.generate(token.input_ids, generation_config=model.generation_config)

        result = tokenizer.batch_decode(generate_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)

        socket.send(pickle.dumps(result[0]))
        print(f"Respond with {result[0]}", flush=True)
