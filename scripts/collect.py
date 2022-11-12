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
from datetime import datetime
from pathlib import Path
import cv2
import os
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats, output_unit=1e-03)


# @profile
def main():
    # load config
    config = load_config()

    # setup car an camera
    print("Setting up car, camera, pid, serial...")

    camera = CSICamera(
        width=int(config["model"]["img_size"]), 
        height=int(config["model"]["img_size"]), 
        capture_fps=int(config["model"]["fps"]),
        capture_device=int(config["model"]["road_cam"])
    )

    ceiling_cam = CSICamera(
        width=int(config["model"]["img_size"]), 
        height=int(config["model"]["img_size"]), 
        capture_fps=int(config["model"]["fps"]),
        capture_device=int(config["model"]["ceiling_cam"])
    )

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
            averaged_speed = moving_average(last_read_speed)

            towrite = int(100*averaged_speed)
            ts = int(time.time()*1000)
            
            fname = f"{ts}_{towrite}.jpg"
            cv2.imwrite(os.path.join(road_dir, fname), new_image)

            # write image w/ timesamp + speed in fname
            
        def handle_ceiling_callback(change):
            # save ceiling image to mcap
            new_image = change['new']
            ts = int(time.time()*1000)
            
            fname = f"{ts}.jpg"
            cv2.imwrite(os.path.join(ceiling_dir, fname), new_image)


        camera.observe(handle_image_callback, names='value')
        ceiling_cam.observe(handle_ceiling_callback, names='value')

        while True:
            print("Reloading config\n")
            old_config = config
            config = load_config()
            print_config_diff(old_config, config)


            now = datetime.now()
            saving_dir = Path("../data/") / now.strftime("%m_%d_%Y_%H_%M_%S")
            ceiling_dir = saving_dir / "ceiling/"
            road_dir = saving_dir / "road/"
            os.makedirs(ceiling_dir)
            os.makedirs(road_dir)
            print(f"making dirs {ceiling_dir}, {road_dir}")

            moving_average = MovingAverage(horizon=int(config["pid"]["moving_average_horizon"]))
            time.sleep(1)

            camera.running = True
            ceiling_cam.running = True

            print("skrrrrt")
            
            prev_value = None
            while True: 
                value = GPIO.input(input_pin)
                if value != prev_value:
                    if value == GPIO.LOW:
                        camera.running = False
                        ceiling_cam.running = False
                        break
                time.sleep(1)

    except KeyboardInterrupt:
        ser.close()
        GPIO.cleanup()


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
