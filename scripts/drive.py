import configparser

from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar
import torch
import PIL.Image
from torch2trt import TRTModule
from torchvision import transforms

import numpy as np
import time
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats, output_unit=1e-03)


def steering_only(output, config):
    x, y = output

    max_speed = float(config["model"]["max_speed"])
    min_speed = float(config["model"]["min_speed"])

    steering = x * float(config["model"]["steering_gain"])
    throttle = min_speed + (max_speed-min_speed) * (1-abs(steering))

    return float(steering), float(throttle)


def e2e(output, config):
    throttle, steering = output
    throttle = float(throttle)
    steering = float(steering)

    steering = steering * float(config["model"]["steering_gain"])

    t = throttle * float(config["model"]["max_speed"])
    accel = throttle > 0
    brake = not accel
    throttle = accel * t + brake * throttle

    return float(steering), float(throttle)


def boost(output, config):
    steering, throttle = output
    throttle = float(throttle)
    steering = -float(steering)

    max_speed = float(config["model"]["max_speed"])
    min_speed = float(config["model"]["min_speed"])

    throttle = max(0, throttle)
    throttle = throttle * (max_speed-min_speed) + min_speed

    return float(steering), float(throttle)


policies = {
    "steering_only": steering_only,
    "e2e": e2e,
    "boost": boost
}


# @profile
def main():
    # load config
    config = load_config()
    model = load_model(config["model"]["model_path"])

    # setup car an camera
    print("Setting up car and camera...")
    car = NvidiaRacecar()

    setup_car(config, car)

    # warmup gpu
    # for _ in range(2):
    #     torch.zeros(1000).to(torch.device('cuda'))

    camera = CSICamera(width=224, height=224, capture_fps=int(config["model"]["fps"]))

    camera.running = True

    def handle_image_callback(change):
        new_image = change['new']
        image = preprocess(new_image).half()
        output = model(image).squeeze()

        car.steering, car.throttle = policies[config["model"]["policy"]](output, config)

    camera.observe(handle_image_callback, names='value')
    print("skrrrrt")


def load_model(path):
    print("Loading model")
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path))
    return model_trt


# takes up the whole RAM if not commented very weird
# mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
# std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


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
