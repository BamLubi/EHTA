import os
import sys

import numpy as np
from tqdm import tqdm


def gen_nov_asv(in_path, out_path, time_slice=30) -> None:
    total_time_slice = int(24 * 60 / time_slice)
    NOV = np.zeros((51, 71, total_time_slice), dtype=np.uint32)
    ASV = np.zeros((51, 71, total_time_slice), dtype=np.float32)
    ASV_cnt = np.zeros((51, 71, total_time_slice), dtype=np.float32)

    for file_name in tqdm(os.listdir(in_path)):
        fi = open(in_path + file_name, "r")
        while True:
            line = fi.readline()
            if not line:
                break
            else:
                line = line.strip("\n").split(",")
                t = int(line[0])
                x = int(line[1])
                y = int(line[2])
                velocity = float(line[3])
                carry = int(line[-1])

                if x >= 0 and x <= 50 and y >= 0 and y <= 70:
                    NOV[x][y][t] += 1
                    if velocity > 0:
                        ASV[x][y][t] += velocity
                        ASV_cnt[x][y][t] += 1
        fi.close()

    ASV = np.divide(ASV, ASV_cnt, out=np.zeros_like(ASV), where=ASV_cnt != 0)
    np.save(out_path + "asv.npy", ASV)
    np.save(out_path + "nov.npy", NOV)


def gen_D(time_slice=30) -> None:
    DATE = ["20150413", "20150414", "20150415",
            "20150416", "20150417", "20150418", "20150419"]
    total_time_slice = int(24 * 60 / time_slice)
    avg_min_nov = 0
    avg_max_nov = 0

    for date in DATE:
        NOV = np.zeros((51, 71, total_time_slice), dtype=np.uint32)
        path = "\\Data\\BlockData\\30\\" + date + "\\"
        for file_name in tqdm(os.listdir(path)):
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

                    if x >= 0 and x <= 50 and y >= 0 and y <= 70:
                        NOV[x][y][t] += 1
            fi.close()
        avg_max_nov += np.max(NOV)
        avg_min_nov += np.min(NOV)
    avg_max_nov /= 7
    avg_min_nov /= 7
    fo = open("\\Experiment\\data.txt", "w")
    fo.writelines("{}: {}\n".format("Average max NOV", avg_max_nov))
    fo.writelines("{}: {}\n".format("Avergae min NOV", avg_min_nov))
    fo.close()


if __name__ == "__main__":
    DATE = ["20150413", "20150414", "20150415",
            "20150416", "20150417", "20150418", "20150419"]
    time_slice = 30

    for date in DATE:
        in_path = "\\Data\\BlockData\\30\\" + date + "\\"
        out_path = "\\Experiment\\npy\\" + date + "_"

        gen_nov_asv(in_path, out_path, time_slice)

        print("Success generate nov and asv of ", date)

    gen_D(time_slice)
    print("Success generate D")
