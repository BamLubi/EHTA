import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from tqdm import tqdm


TOTAL_TAXI = 10095

def draw_7_plot(data: np.ndarray, time_slice=15) -> None:
    x = [i for i in range(int(24 * 60 / time_slice))]
    tags = [str("%02d" % int(i * time_slice / 60)) + ":" + "%02d" %
            int(i * time_slice % 60) for i in x]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)

    ax.plot(x, data[0], linewidth=2, label="Monday")
    ax.plot(x, data[1], linewidth=2, label="Tuesday")
    ax.plot(x, data[2], linewidth=2, label="Wednesday")
    ax.plot(x, data[3], linewidth=2, label="Thursday")
    ax.plot(x, data[4], linewidth=2, label="Friday")
    ax.plot(x, data[5], linewidth=2, label="Saturday")
    ax.plot(x, data[6], linewidth=2, label="Sunday")

    plt.xticks(x, tags, rotation=60)
    ax.set_xlim(0, int(24 * 60 / time_slice))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("Time(2015-04)", fontsize=12)
    ax.set_ylabel("The proportion of vehicles carrying passengers", fontsize=12)

    plt.tight_layout()
    plt.legend()
    plt.show()


def draw_plot(data: np.ndarray, time_slice=15) -> None:
    x = [i for i in range(int(24 * 60 / time_slice))]
    tags = [str("%02d" % int(i * time_slice / 60)) + ":" + "%02d" %
            int(i * time_slice % 60) for i in x]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)

    ax.plot(x, data, c='g', linewidth=2)
    plt.xticks(x, tags, rotation=60)
    ax.set_xlim(0, int(24 * 60 / time_slice))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("time(2015-04-13)", fontsize=12)
    ax.set_ylabel("proportion(%)", fontsize=12)

    plt.title("Vehicular Temporal Distribution Figure")
    plt.tight_layout()
    plt.legend()
    plt.show()


def prepare(path: str, time_slice=15) -> np.ndarray:
    data = [0] * int(24 * 60 / time_slice)
    cnt = [0] * int(24 * 60 / time_slice)

    for file_name in tqdm(os.listdir(path)):
        fi = open(path + file_name, "r", encoding="utf-8")
        tmp_data = [0] * int(24 * 60 / time_slice)
        tmp_cnt = [0] * int(24 * 60 / time_slice)
        while True:
            line = fi.readline()
            if not line:
                break
            else:
                line = line.strip("\n").split(",")
                time = line[1].split(":")
                time = int(time[0]) * 60 + int(time[1])
                time_index = int(time / time_slice)
                if line[-2] != '0.0':
                    tmp_data[time_index] += float(line[-2])
                    tmp_cnt[time_index] += 1
        fi.close()
        data = [x + y for x, y in zip(data, tmp_data)]
        cnt = [x + y for x, y in zip(cnt, tmp_cnt)]

    return np.array(data) / np.array(cnt)


if __name__ == "__main__":
    time_slice = 30
    data = prepare("\\Data\\OriginData\\20150413\\", time_slice)
    npy_path = "\\Analysis\\velocity\\npy\\20150413.npy"
    np.save(npy_path, data) 
    draw_plot(data, time_slice)
    # draw_7_plot(data, time_slice)