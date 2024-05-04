import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def draw_heatmap(data: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    __max = np.max(data)
    data[np.where(data == __max)] = 0.5
    #
    __min = np.min(data)
    __max = np.max(data)
    sns.heatmap((data - __min) / (__max - __min), cmap="YlGnBu", square=True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def prepare(path: str, time_slice = 15) -> np.ndarray:
    # (30.90~31.40)(121.15~121.85)
    data = np.zeros((51, 71), dtype=np.uint32)

    for file_name in tqdm(os.listdir(path)):
        fi = open(path + file_name, "r")
        loc = np.zeros((int(24 * 60 / time_slice), 2), dtype=np.float32)
        cnt = [0] * int(24 * 60 / time_slice)
        while True:
            line = fi.readline()
            if not line:
                break
            else:
                line = line.strip("\n").split(",")
                lng = float(line[2]) + 0.003162
                lat = float(line[3]) - 0.002186
                time_str = line[1].split(":")
                time = int(time_str[0]) * 60 + int(time_str[1])
                idx = int(time / time_slice)
                if idx >= int(24 * 60 / time_slice):
                    print("error!", idx)
                # 统计
                cnt[idx] += 1
                loc[idx][0] += lng
                loc[idx][1] += lat
        fi.close()
        # 计算每个时隙的平均位置
        for i in range(int(24 * 60 / time_slice)):
            if int(loc[i][1]) == 0 or int(loc[i][1]) == 0:
                continue
            x = 3140 - int(loc[i][1] * 100 / cnt[i])
            y = int(loc[i][0] * 100 / cnt[i]) - 12115
            if x >= 0 and x <= 50 and y >= 0 and y <= 70:
                data[x][y] += 1
    
    return data


def draw_100_15_archive():
    npy_path = "\\Analysis\\spatial_dis\\100_15.npy"
    data = np.load(npy_path)
    
    draw_heatmap(data[1:34, 12:49])


if __name__ == "__main__":
    data = prepare("\\Data\\OriginData\\20150415\\", 30)
    npy_path = "\\Analysis\\spatial_dis\\npy\\20150415.npy"
    np.save(npy_path, data)
    data = np.load(npy_path)
    draw_heatmap(data[1:34, 12:49])
    