import sys
sys.path.append("/red/gatortron-phi/gpt/Megatron-LM-3860e99")
from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer

import numpy as np
import time
import os

import argparse
from pathlib import Path


def main(args):
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.model_parallel_size = 1
    
    pin = Path(args.input)
    data_path_prefix = [str(each)[:-4] for each in pin.glob("*.bin")]

    pout = Path(args.output)
    pout.mkdir(parents=True, exist_ok=True)
    output_bin_files = pout / f"{args.output_prefix}.bin"
    output_idx_files = pout / f"{args.output_prefix}.idx"
    try:
        os.remove(output_bin_files)
        os.remove(output_idx_files)
    except:
        pass
    
    tokenizer = build_tokenizer(args)

    builders = indexed_dataset.make_builder(output_bin_files,  impl='mmap', vocab_size=tokenizer.vocab_size)
    for each in data_path_prefix:
        builders.merge_file_(each)

    builders.finalize(output_idx_files) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="input tsv file")
    parser.add_argument("--output", type=str, required=True,
                        help="output path")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="the prefix file ")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="vocab_file full path")
    parser.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    parser.add_argument('--append_eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    parser.add_argument("--tokenizer_type", type=str, default="BertWordPieceCase",
                        help="tokenizer_type")
    global_args = parser.parse_args()
    main(global_args)
