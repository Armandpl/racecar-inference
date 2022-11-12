import serial
from threading import Thread
import time

def main():
    global last_read_speed
    last_read_speed = 0.0

    def receive_speed(ser):
        global last_read_speed

        while True:
            line = ser.readline()   # read a '\n' terminated line
            last_read_speed = float(line.decode("utf-8"))

    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)

    print("Launching serial thread")
    Thread(target=receive_speed, args=(ser,)).start()

    for _ in range(500):
        print(last_read_speed)
        time.sleep(1/40)

if __name__ == "__main__":
    main()