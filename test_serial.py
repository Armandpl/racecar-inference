import serial
import time
import numpy as np
import plotly.express as px
import pandas as pd
from datetime import datetime
from jetracer.nvidia_racecar import NvidiaRacecar
from simple_pid import PID

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats, output_unit=1e-03)



TURN_SPEED = 1
STRAIGHT_SPEED = 1.5

MIN_THROTTLE = -1
MAX_THROTTLE = 0.5
# @profile
def main():
    # TODO remove
    speeds = []
    throttles = []
    targets = []
    flag = True
    ##

    print("setting up car")
    car = NvidiaRacecar()
    car.throttle_gain = 1
    print("car setup")

    pid_speed = PID(0.4, 0.1, 0, setpoint=0)
    pid_speed.output_limits = (MIN_THROTTLE, MAX_THROTTLE)

    with serial.Serial('/dev/ttyACM0', 9600, timeout=1) as ser:
        print("go")
        for i in range(500):
            # line = ser.readline()   # read a '\n' terminated line
            # print(len(line))
            # speed = float(line.decode("utf-8"))
            speed = 0
            time.sleep(30)

            target_speed = flag * TURN_SPEED + (not flag) * STRAIGHT_SPEED
            if i % 200 == 0:
                flag = not flag

            pid_speed.setpoint = target_speed
            throttle = pid_speed(speed)
            car.throttle = throttle

            speeds.append(speed)
            throttles.append(throttle)
            targets.append(target_speed)

            # ser.flushInput()
            # time.sleep(1/40)

    # plotting
    print("plotting")
    speeds = np.array(speeds)
    throttles = np.array(throttles)
    targets = np.array(targets)
    df = pd.DataFrame(list(zip(speeds, throttles, targets)), columns=["actual_speed", "throttle", "target_speed"])
    fig1 = px.line(df, y=["actual_speed", "throttle", "target_speed"])

    now = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    fig1.write_image(f"plots/{now}.png", scale=0.5)

if __name__ == "__main__":
    main()