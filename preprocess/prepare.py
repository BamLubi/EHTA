import os

from tqdm import tqdm

DATE = "20150419"

base_path = "\\Data\\OriginData\\"
origin_path = base_path + DATE + ".txt"
vehicle_path = base_path + DATE + "\\"
tmp_path = base_path + DATE + "_tmp.txt"
log_path = base_path + "readme.txt"


TOTAL_TRAJ = 0
print("0. Count total traj...")
fi = open(origin_path, "r", encoding='utf-8')
while True:
    line = fi.readline()
    if not line:
        break
    else:
        TOTAL_TRAJ += 1
fi.close()
fo = open(log_path, "a")
fo.writelines("TOTAL_TRAJ {} origin is {}\n".format(DATE, TOTAL_TRAJ))
fo.close()
print("Total traj", TOTAL_TRAJ)


fi = open(origin_path, "r", encoding='utf-8')
fo = open(tmp_path, "w")
delete_cnt = 0
print("1. Delete illegal line...")
for i in tqdm(range(TOTAL_TRAJ)):
    line = fi.readline()
    if not line:
        break
    else:
        line = line.strip("\n").split(",")
        t = line[6].split(" ")
        if len(line) != 13 or len(t) != 2:
            delete_cnt += 1
            continue
        try:
            status = 0 if int(line[2]) == 1 else 1
        except ValueError as e:
            delete_cnt += 1
            continue
        fo.writelines("{},{},{},{},{},{}\n".format(line[0], t[1], line[8], line[9], line[-3], status))
TOTAL_TRAJ -= delete_cnt  
fi.close()
fo.close()
print("1.1 Delete origin file and rename...")
os.remove(origin_path)
os.rename(tmp_path, origin_path)
print("Total traj", TOTAL_TRAJ)
fo = open(log_path, "a")
fo.writelines("TOTAL_TRAJ {} valid is {}\n".format(DATE, TOTAL_TRAJ))
fo.close()


print("2. Split all data to single vehicle...")
id_list = {}
fi = open(origin_path, "r", encoding='utf-8')
for i in tqdm(range(TOTAL_TRAJ)):
    line = fi.readline()
    if not line:
        break
    else:
        id = line[0:5]
        if id not in id_list:
            id_list[id] = []
        id_list[id].append(line)
fi.close()
if not os.path.exists(vehicle_path):
    os.makedirs(vehicle_path)
TOTAL_VEHICLE = len(id_list)
cnt = 0
for id, data in id_list.items():
    fo = open(vehicle_path + "{}.txt".format(id), "w")
    MAP = []
    for line in data:
        time = line.strip("\n").split(",")[1].split(":")
        total_time = int(time[0]) * 60 * 60 + int(time[1]) * 60 + int(time[2])
        MAP.append((total_time, line))
    MAP.sort()
    for (x, y) in MAP:
        fo.writelines(y)
    fo.close()
    cnt += 1
    if cnt % 1000 == 0:
        print(str(int(cnt / 1000)) + "/" + str(int(TOTAL_VEHICLE / 1000)))


print("3. Delete illegal vehicle...")
threshold_max = 0.8
threshold_min = 0.2
delete_cnt = 0
for file_name in tqdm(os.listdir(vehicle_path)):
    file_path = vehicle_path + file_name
    fi = open(file_path, "r")
    flg = 0
    passengers = 0
    total = 0
    while True:
        line = fi.readline()
        if not line:
            break
        else:
            line = line.strip("\n").split(",")
            hour = int(line[1].split(":")[0])
            flg |= (1 << hour)
            passengers += int(line[-1])
            total += 1
    rate = passengers / total
    if flg != (1 << 24) - 1 or rate > threshold_max or rate < threshold_min:
        fi.close()
        delete_cnt += 1
        os.remove(file_path)
    else:
        fi.close()
print("Delete", delete_cnt)

print("{} Complete!".format(origin_path))
os.remove(origin_path)