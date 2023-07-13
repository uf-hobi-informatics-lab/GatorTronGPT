import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import json
import random


def select_sampels(ifn, ofd, n, restrict_len=0):
    print("load...")
    with open(ifn, "r") as f:
        lines = f.readlines()
        
    ll = len(lines)
    start = np.random.randint(0, ll-50000)
    lines = lines[start: start+49999]
    
    print("convert to array...")
    lines = np.array(lines)
        
    # select 1/10 of the total samples
    print("get subsamples...")
    ii = np.random.randint(0, lines.shape[0], (lines.shape[0]//10,))
    lines = lines[ii]
    
    if restrict_len:
        lines = [e for e in lines if len(e.split()) >= restrict_len]
        lines = np.array(lines)
    
    print("get pos neg samples...")
    L = lines.shape[0]
    p_idxs = np.random.randint(0, L, (n,))
#     n_idxs = np.asarray([e for e in np.random.randint(0, L, (3*n,)) if e not in set(p_idxs)][:100])
    
    lines = np.array(lines)
    p_samples = lines[p_idxs]
    
    n_samples = set(lines) - set(p_samples)
    
    neg = [json.loads(each)['text'].replace("\n", " ") for each in n_samples]
    with open(f"{ofd}/neg.txt", "w") as f:
        f.write("\n".join(neg))
    
    print("prompts...")
    samples = []
    ori_text = []
    for line in p_samples:
        text = json.loads(line)['text'].replace("\n", " ")
        ori_text.append(text)
        new_text = " ".join(text.split(" ")[:random.randint(10, 30)])
        samples.append(new_text)
        
    pst = []
    for pp, tt in zip(samples, ori_text):
        ntt = tt.replace(pp, "").strip()
        pst.append(ntt)
        
    with open(f"{ofd}/new_wiki_{n}.txt", "w") as f:
        f.write("\n".join(samples))
    with open(f"{ofd}/pos.txt", "w") as f:
        f.write("\n".join(pst))
    print("done...")


def to_file(data, fn):
    with open(fn, "w") as f:
        f.write("\n".join(data))


def main(args):
    p = Path(args.output)
    p.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    with open(args.input, "r") as f:
        lines = f.readlines()
    
    # these two choice steps are slow if there are many lines in the file
    selected_lines = np.random.choice(lines, size=args.nprompt, replace=False)
    negative_lines = np.random.choice([each for each in lines if each not in selected_lines], size=args.nprompt, replace=False)

    prompts = []
    ori_text = []
    for line in selected_lines:
        text = json.loads(line)['text'].replace("\n", " ")
        ori_text.append(text)
        new_text = " ".join(text.split(" ")[:random.randint(10, 30)])
        prompts.append(new_text)
    
    neg_text = [json.loads(line)['text'].replace("\n", " ") for line in negative_lines]

    to_file(prompts, p/"prompts.txt")
    to_file(ori_text, p/"pos.txt")
    to_file(neg_text, p/"neg.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                       help="input data as jsonl")
    parser.add_argument("--key", type=str, default="text",
                       help="key used to extract text from jsonl")
    parser.add_argument("--seed", type=int, default=13,
                       help="random seed")
    parser.add_argument("--nprompt", type=int, default=100,
                       help="number of prompts to create")
    parser.add_argument("--output", type=str, default=None,
                       help='dir to save prompts, pos samples and neg samples')
    args = parser.parse_args()
    main(args)
