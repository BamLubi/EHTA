import os
import sys

import numpy as np
from tqdm import tqdm


def prepare(in_path: str, out_path: str, time_slice=15) -> None:
    # (30.90~31.40)(121.15~121.85)
    for file_name in tqdm(os.listdir(in_path)):
        fi = open(in_path + file_name, "r")
        fo = open(out_path + file_name, "w")

        loc = np.zeros((int(24 * 60 / time_slice), 2), dtype=np.float32)
        cnt = [0] * int(24 * 60 / time_slice)
        carry = [0] * int(24 * 60 / time_slice)
        velocity = [0] * int(24 * 60 / time_slice)
        velocity_cnt = [0] * int(24 * 60 / time_slice)

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
                cnt[idx] += 1
                loc[idx][0] += lng
                loc[idx][1] += lat
                if line[-1] == "1":
                    carry[idx] = 1
                if line[-2] != '0.0':
                    velocity[idx] += float(line[-2])
                    velocity_cnt[idx] += 1
        fi.close()
        velocity = np.array(velocity, dtype=float)
        velocity_cnt = np.array(velocity_cnt, dtype=float)
        velocity = np.divide(velocity, velocity_cnt, out=np.zeros_like(velocity), where=velocity_cnt!=0)
        np.true_divide
        for i in range(int(24 * 60 / time_slice)):
            if int(loc[i][1]) == 0 or int(loc[i][1]) == 0:
                continue
            x = 3140 - int(loc[i][1] * 100 / cnt[i])
            y = int(loc[i][0] * 100 / cnt[i]) - 12115
            if x < 0 or x > 50 or y < 0 or y > 70:
                x = y = -1
            fo.writelines("{},{},{},{},{}\n".format(i, x, y, velocity[i], carry[i]))


if __name__ == "__main__":
    DATE = sys.argv[1]
    in_path = "\\Data\\OriginData\\" + DATE + "\\"
    out_path = "\\Data\\BlockData\\30\\" + DATE + "\\"
    time_slice = 30
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    prepare(in_path, out_path, time_slice)
