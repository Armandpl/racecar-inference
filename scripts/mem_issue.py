import configparser

from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar
import torch
import PIL.Image
from torch2trt import TRTModule
from torchvision import transforms

import numpy as np
import time

def main():
    # load config
    config = load_config()
    # model = load_model(config["model"]["model_path"])

    # setup car an camera
    print("Setting up car and camera...")
    time.sleep(60)
    car = NvidiaRacecar()

    setup_car(config, car)

    # warmup gpu
    for _ in range(2):
        torch.zeros(1000).cuda()

    camera = CSICamera(width=224, height=224, capture_fps=int(config["model"]["fps"]))

    camera.running = True

    def handle_image_callback(change):
        new_image = change['new']
        image = preprocess(new_image).half()
        output = model(image).squeeze()

        car.steering, car.throttle = policies[config["model"]["policy"]](output, config)

    # camera.observe(handle_image_callback, names='value')
    print("time sleep")
    time.sleep(60)
    print("skrrrrt")


def load_model(path):
    print("Loading model")
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path))
    return model_trt




def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    # image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def setup_car(config, car):
    car.steering_gain = float(config["car"]["steering_gain"])
    car.steering_offset = float(config["car"]["steering_offset"])
    car.throttle_gain = float(config["car"]["throttle_gain"])


def load_config(config_path="config.ini"):
    print("Loading config")
    config = configparser.ConfigParser()
    config.read(config_path)

    return config


if __name__ == "__main__":
    main()
