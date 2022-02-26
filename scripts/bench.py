import argparse

import torch
from torch2trt import TRTModule

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats, output_unit=1e-03)

@profile
def main(args):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.model_path))
    model_trt.cuda()
    model_trt.eval()

    data = torch.ones((1, 224, 224, 3))

    # warmup gpu
    for _ in range(2):
        torch.zeros(1000).cuda()

    for _ in range(200):
        model_trt(data.cuda().half())

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        type=str
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
