import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


z = np.zeros((50, 50, 70))

path = "\\Data\\Block_Data_15\\20150413\\"
count = 0
for file_name in tqdm(os.listdir(path)):
    if count == 50:
        break
    fi = open(path + file_name, "r")
    while True:
        line = fi.readline()
        if not line:
            break
        else:
            line = line.strip("\n").split(",")
            x = int(line[1])
            y = int(line[2])
            if x == -1 or y == -1:
                continue
            z[count][x][y] += 1
    count += 1

# 画热力图
for i in range(50):
    plt.figure(figsize=(20, 10))
    sns.heatmap(z[i], annot=True, cmap='Blues', square=True)
    plt.title("vehicle {}".format(i))
    plt.savefig("\\Analysis\\50_vehicle_traj_heatmap\\{}.png".format(i))
    plt.close()