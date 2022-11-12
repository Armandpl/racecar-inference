import argparse
import logging
import os

import torch
import torchvision
import wandb

from torch2trt import torch2trt
from new_model import RoadRegressionTurn

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats, output_unit=1e-03)

def main():
    print("loading model")
    # model_name = "squeezenet1_0"
    # model = torchvision.models.__dict__[model_name]()
    # model.classifier = torch.nn.Linear(in_features=86528, out_features=2)
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    # model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        
    # gpunet.to(device)
    # gpunet.eval()
    # model = RoadRegressionTurn()
    model.load_state_dict(torch.load("models/unoptimized/resnet18-normal.pth"))
    print("model loaded")
    model.cuda().eval().half()
    data = torch.zeros((1, 3, 160, 160)).cuda().half()
    print("converting")
    model_trt = torch2trt(model, [data], fp16_mode=True)
    print("saving")
    # torch.save(model_trt.state_dict(), f"models/dummy_{model_name}.pth")
    torch.save(model_trt.state_dict(), f"models/road4.pth")


if __name__ == "__main__":
    main()
