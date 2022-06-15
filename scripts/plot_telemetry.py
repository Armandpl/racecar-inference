import numpy as np
import plotly.express as px
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    print("loading data")
    data =  np.load('telemetry.npy', allow_pickle=True)
    print("making df")
    df = pd.DataFrame(list(zip(data[0], data[1], data[2])), columns=["actual_speed", "throttle", "target_speed"])
    print("making plot")

    # gca stands for 'get current axis'
    ax = plt.gca()

    df.plot(kind='line',y='actual_speed', ax=ax)
    df.plot(kind='line',y='throttle', color='red', ax=ax)
    df.plot(kind='line',y='target_speed', color='green', ax=ax)

    print("saving fig")
    now = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    plt.savefig(f"plots/{now}.jpeg")

    plt.show()

    # fig1 = px.line(df, y=["actual_speed", "throttle", "target_speed"])

    # now = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    # print("saving fig")
    # fig1.write_image(f"plots/{now}.jpeg", scale=0.5)

if __name__ == "__main__":
    main()