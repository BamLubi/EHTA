import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


z = np.zeros((96, 50, 70))

path = "\\Data\\Block_Data_15\\20150413\\"
count = 0
for file_name in tqdm(os.listdir(path)):
    if count == 1000:
        break
    fi = open(path + file_name, "r")
    while True:
        line = fi.readline()
        if not line:
            break
        else:
            line = line.strip("\n").split(",")
            t = int(line[0])
            x = int(line[1])
            y = int(line[2])
            if x == -1 or y == -1:
                continue
            z[t][x][y] += 1
    count += 1

# 画热力图
for i in range(96):
    plt.figure(figsize=(20, 10))
    sns.heatmap(z[i], annot=True, cmap='Blues', square=True)
    plt.title("%02d" % int(i * 15 / 60) + ":%02d" % int(i * 15 % 60))
    plt.savefig("\\Analysis\\traj_heatmap\\{}.png".format(i))
    plt.close()