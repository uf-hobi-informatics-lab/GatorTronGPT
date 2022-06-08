# -*- coding: utf-8 -*-

import sys
sys.path.append("/red/gatortron-phi/gpt/Megatron-LM-latest")
sys.path.append("/red/gatortron-phi/gpt/Megatron-LM-latest/megatron")

from megatron import get_args
from megatron import print_rank_0
# from megatron import get_tokenizer
# from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel, ModelType
from megatron.training import get_model
from megatron.text_generation_utils import generate
import numpy as np
import datetime
import json
import re
from tqdm import tqdm
import traceback


def load_conditions(fn, mbs=8):
    with open(fn, "r") as f:
        lines = f.readlines()

    split_num = len(lines) // mbs + 1

    return np.array_split(lines, split_num), lines


def write_to_file(data, fn):
    mode = "w"

    data = [each.replace("\n", " ") for each in data]

    text_to_write = "\n".join(data)

    with open(fn, mode=mode, encoding="utf-8") as f:
        f.write(text_to_write)


def write_to_json(gens, fn):
    with open(fn, "w", encoding="utf-8") as f:
        for text in gens:
            text = re.sub(" +", " ", text)
            f.write(json.dumps({"text": text}))
            f.write("\n")


# def write_to_json(prompts, gens, fn):
#     prps = []
#     for each in prompts:
#         prps.extend(list(each))

#     d = dict()
#     for p, g in zip(prps, gens):
#         d[p] = g

#     with open(fn, "w") as f:
#         json.dump(d, f, indent=2)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=64,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--data_split", type=int, default=4,
                       help='how many iterations used to feed all condition sentences')
    group.add_argument('--keep_prompt', type=int, default=0,
                       help='keep prompts in generated texts')
    group.add_argument('--to_json', type=int, default=0,
                       help='output will be in json format for megatron training')
    #eos_id

    return parser


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(model_provider, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Generate samples.
    condition_samples, prompts = load_conditions(args.sample_input_file, args.micro_batch_size)

    print_rank_0("start generating text...")
    print_rank_0(f"start time: {datetime.datetime.now()}")
    outputs = []
    for samples in tqdm(condition_samples):
        try:
            resp_sentences, _, _, _, _ = generate(
                model, sentences=samples, tokens_to_generate=args.out_seq_length, temperature=args.temperature)
            # remove all content after <|endoftext|>
            resp_sentences = [each.split("<|endoftext|>")[0] for each in resp_sentences]
            outputs.extend(resp_sentences)
        except Exception as ex:
            print_rank_0(samples)
            print_rank_0(samples.shape)
            print_rank_0(traceback.format_exc())

    if args.keep_prompt:
        formatted_outputs = outputs
    else:
        formatted_outputs = []
        for gt, st in zip(outputs, prompts):
            ngt = gt.replace(st, "").strip()
            formatted_outputs.append(ngt)

    if args.to_json:
        write_to_json(formatted_outputs, args.sample_output_file)
    else:
        write_to_file(formatted_outputs, args.sample_output_file)
    

    print_rank_0(f"end time: {datetime.datetime.now()}")


if __name__ == "__main__":
    main()