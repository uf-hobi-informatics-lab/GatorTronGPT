import argparse
import os
import random
import json
import math


def load_lines(fn):
    with open(fn, "r") as f:
        lines = f.readlines()
    return lines


def main(args):
    extracted_samples = []
    word_cnt = 0
    word_cnts = []
    note_cnt = 0

    files = [f for f in os.listdir(args.input) if f.endswith(".json")]

    for fn in files:
        lines = load_lines(args.input + "/" + fn)
        idx = [random.randint(0, len(lines)-1) for _ in range(args.num)]
        for i in idx:
            choiced = lines[i]
            choiced = json.loads(choiced)['text']
            choiced = choiced.encode('utf-8')
            word_cnt += len(choiced.split(" "))
            word_cnts.append(len(choiced.split(" ")))
            note_cnt += 1
            extracted_samples.append(choiced)
    
    print ("average word: ", word_cnt/note_cnt)
    print ("average word (std): ", math.sqrt(sum([(e-(word_cnt/note_cnt))**2 for e in word_cnts])/note_cnt))

    with open(args.output, "w") as f:
        f.write("\n".join(extracted_samples))

# python scripts/random_extract_samples.py 
# --input gpt_syn/gpt_results/gpt_20b_large_scale_gen_mimic_700_1.2_0.9_512/ 
# --output sample700512.json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="where to draw random samples, a directory")
    parser.add_argument("--num", type=int, default=1, 
                        help="number of samples to extract from each file")
    parser.add_argument("--output", type=str, required=True,
                        help="output file")
    global_args = parser.parse_args()
    main(global_args)
    