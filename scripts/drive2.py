import configparser

from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar
import torch
import PIL.Image
from torch2trt import TRTModule
from torchvision import transforms

import numpy as np

import RPi.GPIO as GPIO
import time
from simple_pid import PID
import serial

from threading import Thread

from utils.moving_average import MovingAverage
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats, output_unit=1e-03)
TELEMETRY_BUFFER = 5000


def steering_only(output, config):
    x, y = output

    max_speed = float(config["model"]["max_throttle"])
    min_speed = float(config["model"]["min_throttle"])

    steering = x * float(config["model"]["steering_gain"])
    throttle = min_speed + (max_speed-min_speed) * (1-abs(steering))


    return float(steering), float(throttle)


def speed_policy(output, config):
    x, y = output

    turn_speed = float(config["pid"]["turn_speed"])
    straight_speed = float(config["pid"]["straight_speed"])

    steering = x * float(config["model"]["steering_gain"])

    if abs(x) < float(config["pid"]["dead_zone"]):
        target_speed = straight_speed
    else:
        target_speed = turn_speed + (straight_speed-turn_speed) * (1-abs(x))

    return float(steering), float(target_speed)


policies = {
    "steering_only": steering_only,
    "speed_policy": speed_policy,
}


# @profile
def main():
    # load config
    config = load_config()
    model = load_model(config["model"]["model_path"])

    # setup car an camera
    print("Setting up car, camera, pid, serial...")
    car = NvidiaRacecar()

    setup_car(config, car)

    speed_pid = PID(0, 0, 0)
    setup_pid(config, speed_pid)

    # warmup gpu
    for _ in range(2):
        torch.zeros(1000).cuda()

    camera = CSICamera(
        width=int(config["model"]["img_size"]), 
        height=int(config["model"]["img_size"]), 
        capture_fps=int(config["model"]["fps"]),
        capture_device=int(config["model"]["road_cam"])
    )

    # ceiling_cam = CSICamera(
    #     width=int(config["model"]["img_size"]), 
    #     height=int(config["model"]["img_size"]), 
    #     capture_fps=int(config["model"]["fps"]),
    #     capture_device=int(config["model"]["ceiling_cam"])
    # )

    ser_device = config["serial"]["device"]
    ser_baudrate = int(config["serial"]["baudrate"])
    ser_timeout = float(config["serial"]["timeout"])

    global last_read_speed
    last_read_speed = 0.0
    moving_average = MovingAverage(horizon=int(config["pid"]["moving_average_horizon"]))

    def receive_speed(ser):
        global last_read_speed

        while True:
            try:
                line = ser.readline()   # read a '\n' terminated line
                last_read_speed = float(line.decode("utf-8"))
            except Exception as e:
                print(e)
                last_read_speed = None

    ser = serial.Serial(ser_device, ser_baudrate, timeout=ser_timeout)

    print("Launching serial thread")
    Thread(target=receive_speed, args=(ser,)).start()

    # setup button
    input_pin = "DAP4_SCLK"  # BCM pin 18, BOARD pin 12
    # GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin

    try:
        def handle_image_callback(change):
            # global last_read_speed
            new_image = change['new']
            image = preprocess(new_image).half()
            output = model(image).squeeze()

            if config["model"]["policy"] == "steering_only":
                car.steering, car.throttle = policies[config["model"]["policy"]](output, config)
            elif config["model"]["policy"] == "speed_policy":
                steering, target_speed = policies[config["model"]["policy"]](output, config)

                # target_speed = turn_speed * flag + (not flag) * straight_speed
                ###

                if last_read_speed is not None:
                    averaged_speed = moving_average(last_read_speed)
                    speed_pid.setpoint = target_speed
                    throttle = speed_pid(averaged_speed)
                else:
                    throttle = float(config["pid"]["fallback_throttle"])

                car.steering, car.throttle = steering, throttle
            else:
                print("wrong policy name")
                time.sleep(1)
        
        # def handle_ceiling_callback(change):
        #     # save ceiling image to mcap

        #     # might as well read the IMU here
        #     pass

        camera.observe(handle_image_callback, names='value')
        # ceiling_cam.observe(handle_ceiling_callback, names='value')

        while True:
            print("Reloading config\n")
            old_config = config
            config = load_config()
            print_config_diff(old_config, config)

            setup_car(config, car)
            speed_PID = PID(0, 0, 0)
            speed_PID.setpoint = 0
            setup_pid(config, speed_pid)
            moving_average = MovingAverage(horizon=int(config["pid"]["moving_average_horizon"]))
            time.sleep(1)

            camera.running = True
            # ceiling_cam.running = True

            print("skrrrrt")
            
            prev_value = None
            while True: 
                value = GPIO.input(input_pin)
                if value != prev_value:
                    if value == GPIO.LOW:
                        camera.running = False
                        # ceiling_cam.running = False
                        break
                time.sleep(1)

    except KeyboardInterrupt:
        ser.close()
        GPIO.cleanup()


def load_model(path):
    print("Loading model")
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(path))
    return model_trt


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


def setup_pid(config, pid):
    p = float(config["pid"]["p"])
    i = float(config["pid"]["i"])
    d = float(config["pid"]["d"])
    pid.tunings = (p, i, d)

    min_throttle = float(config["pid"]["min_throttle"])
    max_throttle = float(config["pid"]["max_throttle"])
    pid.output_limits = (min_throttle, max_throttle)

    # pid.sample_time = 1/float(config["model"]["fps"])

def load_config(config_path="config.ini"):
    print("Loading config")
    config = configparser.ConfigParser()
    config.read(config_path)

    return config


def print_config_diff(old_config, new_config):
    old_dict = old_config._sections
    new_dict = new_config._sections
    print("New values:")
    for i, j in zip(old_dict.items(), new_dict.items()):
        if j != i:
            print(type(j))
            print(j)
            


if __name__ == "__main__":
    main()
