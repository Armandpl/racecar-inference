import argparse
import logging
import os

import torch
import torchvision
import wandb

from torch2trt import torch2trt

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats, output_unit=1e-03)


def convert(model_pth, model_architecture, model_out_dims):
    logging.info("Creating model architecture")
    model = torchvision.models.__dict__[model_architecture](pretrained=False)

    model.fc = torch.nn.Linear(model.fc.in_features, model_out_dims)
    model = model.cuda().eval().half()

    model.load_state_dict(torch.load(model_pth))

    # dummy input
    data = torch.zeros((1, 3, 224, 224)).cuda().half()

    logging.info("Optimizing model...")
    model_trt = torch2trt(model, [data], fp16_mode=True)

    return model_trt


@profile
def main(args):
    with wandb.init(
        project=args.project,
        config=args,
        job_type="trt-optimization",
        entity=args.entity
    ) as run:

        logging.info("Downloading non optimized model")
        artifact = run.use_artifact(f"model:{args.version}")
        artifact_dir = artifact.download()

        # fetching the model architecture from the producer run
        producer_run = artifact.logged_by()
        # model_architecture = producer_run.config["architecture"]
        model_architecture = "resnet18"
        model_pth = os.path.join(artifact_dir, "model.pth")

        model_trt = convert(model_pth, model_architecture, 2)

        # bench
        data = torch.zeros((1, 3, 224, 224))
        for _ in range(100):
            model_trt(data.cuda().half())

        # save the model for inference
        logging.info("Saving model")
        torch.save(model_trt.state_dict(), args.output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull the latest trained model, optimize it and log it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Where to save the model"
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="Artifact version"
    )
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        default="wandb",
        help="Which entity owns the project. None = you"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="racecar",
        help="Project the dataset belongs to."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
