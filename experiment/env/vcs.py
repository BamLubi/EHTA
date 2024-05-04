from __future__ import annotations

import math
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from env.location import Location
from env.task import Task
from env.vehicle import Vehicle
from tqdm import tqdm

CITY_M = 51
CITY_N = 71
MAX_TIME_SLICE = 48
MIN_TIME_SLICE = 0
HIS_NOV_MIN = 0 # history min number of nov
HIS_NOV_MAX = 365 # history max number of nov
SPEED_LEVEL = 100
TASK_TYPE = ["discrete", "continuous", "periodic"]
COPY_VEHICLE = False

N_VEHICLE = 10
N_TASK = 500

EHTA_PATH = "项目文件夹\\"
EXP_PATH = EHTA_PATH + "experiment"
VEHICLE_PATH = EXP_PATH + "data\\"


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def copy_vehicle() -> None:
    path = EHTA_PATH + "Data\\Block_Data_30\\20150419\\"
    new_path = VEHICLE_PATH + str(N_VEHICLE) + "\\"
    cnt = 0
    for file_name in os.listdir(path):
        if cnt >= N_VEHICLE:
            break
        if np.random.uniform() > 0.1:
            continue
        fi = open(path + file_name, "r")
        t = 0
        is_wrong = False
        while True:
            line = fi.readline()
            if not line:
                break
            else:
                line = line.strip("\n").split(",")
                if int(line[0]) != t:
                    is_wrong = True
                    break
                t += 1
        fi.close()
        if is_wrong == False:
            cnt += 1
            shutil.copyfile(path + file_name, new_path + file_name)
    print("COPY SUCC")

class VCS(object):
    def __init__(self, n_vehicle=10, malicious=0, alg_flg=0, malicious_flg=0, seed=2023, task_type="uniform", vehicle_set="vechiles_1", fix_time=0) -> None:
        global N_VEHICLE, VEHICLE_PATH
        N_VEHICLE = n_vehicle
        
        np.random.seed(seed)
        
        VEHICLE_PATH = EXP_PATH + "data\\" + vehicle_set+ "\\"
        
        self.alg_flg = alg_flg # 0代表普通算法。 1代表我们提出的算法
        self.malicious_flg = malicious_flg # 0代表不开启恶意用户判断
        self.malicious = malicious
        self.vehicles = self.__gen_vehicle()
        self.n_type = [0, 0, 0]
        self.tasks = self.__gen_task(task_type, fix_time) # uniform discrete continuous  periodic
        self.cur_nov_map = np.load(EXP_PATH + "npy\\20150419_nov.npy")
        self.cur_asv_map = np.load(EXP_PATH + "npy\\20150419_asv.npy")
        # EHTA成功识别恶意用户的占比
        self.EHTA_SUCC_MAL = 0

    def __gen_vehicle(self) -> list[Vehicle]:
        if COPY_VEHICLE == True:
            copy_vehicle()
        vehicles = []
        path = VEHICLE_PATH + str(N_VEHICLE) + "\\"
        idx = 1
        for file_name in os.listdir(path):
            fi = open(path + file_name, "r")
            traj = []
            while True:
                line = fi.readline()
                if not line:
                    break
                else:
                    line = line.strip("\n").split(",")
                    traj.append([int(line[0]), int(line[1]), int(
                        line[2]), float(line[3]), int(line[4])])
            fi.close()
            if np.random.uniform() < self.malicious:
                # 恶意用户
                vehicles.append(Vehicle(idx, traj=traj, type=1))
            else:
                vehicles.append(Vehicle(idx, traj=traj))
            idx += 1
        return vehicles

    def __gen_task(self, task_type="uniform", fix_time=0) -> list[Task]:
        tasks = []
        
        # 位置
        x_list = np.random.randint(0, CITY_M, N_TASK)
        y_list = np.random.randint(0, CITY_N, N_TASK)
        
        # 生成符合泊松分布的到达时间
        at_list = np.random.poisson(lam=21, size=N_TASK)
        at_list = np.floor(at_list) % MAX_TIME_SLICE
        # 均匀分布到达时间
        # at_list = np.random.randint(MIN_TIME_SLICE, MAX_TIME_SLICE, N_TASK) 
        
        # 任务类型比例
        # 1. 均匀分布
        type_list = []
        if task_type == "uniform":
            type_list = np.random.randint(0, 3, N_TASK)
        # 2. 仅有一种
        elif task_type == "discrete":
            type_list = np.array([0] * N_TASK)
        elif task_type == "continuous":
            type_list = np.array([1] * N_TASK)
        elif task_type == "periodic":
            type_list = np.array([2] * N_TASK)
        
        # 创建任务
        for idx in range(0, N_TASK):
            type = type_list[idx]
            self.n_type[type] += 1
            # 随机时间
            tasks.append(Task(idx + 1, Location(x_list[idx], y_list[idx]), at_list[idx], TASK_TYPE[type]))
            # 固定时间
            # tasks.append(Task(idx + 1, Location(x_list[idx], y_list[idx]), 40, TASK_TYPE[type]))
            # 固定时间和区域
            # tasks.append(Task(idx + 1, Location(8, 28), fix_time, TASK_TYPE[type]))
            # tasks.append(Task(idx + 1, Location(x_list[idx], y_list[idx]), fix_time, TASK_TYPE[type]))
        return tasks

    @staticmethod
    def __gen_map() -> list[list[int]]:
        return [[0 for j in range(CITY_N)] for i in range(CITY_M)]
    
    def run(self, allocation_alg: function) -> None:
        total_malicious = 0
        EHTA_succ_malicious = 0
        # for time_slice in tqdm(range(MIN_TIME_SLICE, MAX_TIME_SLICE)):
        for time_slice in range(MIN_TIME_SLICE, MAX_TIME_SLICE):
            tasks = self.get_avl_tasks(time_slice)
            vehicles = self.get_avl_vehicles(time_slice)
            decision = allocation_alg(vehicles, tasks, time_slice)
            for i in range(0, len(decision)):
                if decision[i] == 0 or self.can_sense(vehicles[i], decision[i], time_slice) == False or np.random.uniform() < 0.1: # 用户有10%的概率不完成任务
                    continue
                
                # 确定奖励
                r, is_malious_predict = self.get_reward(vehicles[i], decision[i], time_slice, 0)
                
                # 当前车辆的真实身份
                is_malious_true = self.vehicles[vehicles[i] - 1].type
                # 预测为恶意用户
                if self.malicious_flg == 1 and is_malious_predict == 1:
                    continue
                if self.malicious_flg == 1:
                    total_malicious += 1
                    if is_malious_predict == is_malious_true:
                        EHTA_succ_malicious += 1
                
                # 确认分配
                self.vehicles[vehicles[i]-1].sense(decision[i], time_slice, r, self.get_DC(vehicles[i], decision[i]))
                self.tasks[decision[i]-1].sense(vehicles[i], time_slice)
        
        # 确认成功识别恶意用户的占比
        self.EHTA_SUCC_MAL = EHTA_succ_malicious / total_malicious if total_malicious != 0 else 1
        if self.alg_flg == 1:
            print("EHTA_SUCC_MAL", EHTA_succ_malicious, "/", total_malicious, "=", round(self.EHTA_SUCC_MAL, 4))
    
    def pre_sense(self, vehicle_id, task_id, sensed_time) -> float:
        self.tasks[task_id-1].sense(vehicle_id, sensed_time)
        
        # r, _ = self.get_reward(vehicle_id, task_id, sensed_time, 1)
        # self.vehicles[vehicle_id-1].sense(task_id, sensed_time, r, self.get_DC(vehicle_id, task_id))
    
    def pre_desense(self, vehicle_id, task_id, sensed_time) -> float:
        self.tasks[task_id-1].de_sense(vehicle_id, sensed_time)
        
        # r, _ = self.get_reward(vehicle_id, task_id, sensed_time, 1)
        # self.vehicles[vehicle_id-1].de_sense(task_id, sensed_time, r, self.get_DC(vehicle_id, task_id))
    
    def can_sense(self, vehicle_id, task_id, sensed_time) -> bool:
        return self.vehicles[vehicle_id - 1].is_free(sensed_time) and self.tasks[task_id - 1].is_free(sensed_time)

    def get_avl_tasks(self, time_slice: int) -> list[int]:
        tasks_id = []
        for task in self.tasks:
            if task.is_free(time_slice) == True:
                tasks_id.append(task.id)
        return tasks_id

    def get_avl_vehicles(self, time_slice: int) -> list[int]:
        vehicles_id = []
        for vehicle in self.vehicles:
            if vehicle.is_free(time_slice) == True:
                vehicles_id.append(vehicle.id)
        return vehicles_id
    
    def get_DC(self, vehicle_id, task_id) -> int:
        return self.vehicles[vehicle_id-1].loc - self.tasks[task_id-1].loc
    
    def get_reward(self, vehicle_id, task_id, sensed_time, flg) -> tuple:
        # DC
        DC = self.get_DC(vehicle_id, task_id)
        DC = 1 if DC == 0 else DC
        
        # TC
        vehicle_loc = self.vehicles[vehicle_id-1].loc
        v = self.get_ASV(vehicle_loc.x, vehicle_loc.y, sensed_time)
        v = SPEED_LEVEL / 4 if v == 0 else v
        TC = 2 * DC / v
        TC = 1 if TC <= 1 else TC
        
        N_TC = TC
        N_DC = DC
        MAILIOUS = self.vehicles[vehicle_id - 1].type
        if self.malicious_flg == 1 and MAILIOUS == 1:
            N_TC += np.random.poisson(TC, 2)[1]
            N_DC += np.random.poisson(DC, 2)[1]
            # N_TC *= 2
            # N_DC *= 2
 
        if self.alg_flg == 0:
            R = N_TC + N_DC
            return round(R, 4), 0

        if self.alg_flg == 1:
            ## RCI
            min_rci = self.get_min_RCI(vehicle_id, task_id, sensed_time) / DC
            max_rci = self.get_max_RCI(vehicle_id, task_id, sensed_time) / DC
            ## EC
            EC = math.exp(min_rci - 0.6) * (TC + DC)
            MECT = math.exp(max_rci - 0.6) * (TC + DC)
            if self.malicious_flg == 1:
                if N_TC + N_DC >= MECT:
                    return 0, 1
            R = EC
            if flg == 0:
                o_fi, _, _ = self.get_FI()
                self.pre_sense(vehicle_id, task_id, sensed_time)
                n_fi, _, _ = self.get_FI()
                self.pre_desense(vehicle_id, task_id, sensed_time)
                R = (1 +  (n_fi - o_fi) * 100) * EC

            return round(R, 4), 0

        return 0

    def get_FI(self) -> tuple:
        """获取时空公平性指标

        Returns:
            tuple: FI, SF, TF
        """
        # 1. 所有任务的等待时间 -- TF
        a = 0
        b = 0
        for task in self.tasks:
            wt = task.get_waiting_time(MAX_TIME_SLICE)
            a += wt
            b += wt ** 2
        TF = (a ** 2) / (len(self.tasks) * b) if b != 0  else 0

        # 2. 所有任务的数据量 -- SF
        a = 0
        b = 0
        for task in self.tasks:
            wt = task.data
            a += wt
            b += wt ** 2
        SF = (a ** 2) / (len(self.tasks) * b) if b != 0 else 0
        
        FI = 0.5 * TF + 0.5 * SF
        
        return FI, SF, TF
    
    def get_min_RCI(self, vehicle_id: int, task_id: int, sensed_time: int) -> float:
        st_loc: Location = self.vehicles[vehicle_id-1].loc
        ed_loc: Location = self.tasks[task_id-1].loc
        
        if st_loc == ed_loc:
            return self.get_RCI(st_loc.x, st_loc.y, sensed_time)
        
        if st_loc.x >= ed_loc.x and st_loc.y <= ed_loc. y:
            x = st_loc.x - ed_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(0, y):
                    rci = self.get_RCI(ed_loc.x + i , st_loc.y + j, sensed_time)
                    if j == 0 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif i == x-1 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i][j-1], dp[i+1][j])
            return dp[0][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y <= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0, x):
                for j in range(0, y):
                    rci = self.get_RCI(st_loc.x + i, st_loc.y + j, sensed_time)
                    if i == 0 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif j == 0 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i][j-1], dp[i-1][j])
            return dp[x-1][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y >= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0,  x):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(st_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == 0 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i-1][j], dp[i][j+1])
            return dp[x-1][0]
        elif st_loc.x >= ed_loc.x and st_loc.y >= ed_loc.y:
            x = st_loc.x - ed_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(ed_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == x-1 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + min(dp[i+1][j], dp[i][j+1])
            return dp[0][0]

    def get_max_RCI(self, vehicle_id: int, task_id: int, sensed_time: int) -> float:
        st_loc: Location = self.vehicles[vehicle_id-1].loc
        ed_loc: Location = self.tasks[task_id-1].loc
        
        if st_loc == ed_loc:
            return self.get_RCI(st_loc.x, st_loc.y, sensed_time)
        
        if st_loc.x >= ed_loc.x and st_loc.y <= ed_loc. y:
            x = st_loc.x - ed_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(0, y):
                    rci = self.get_RCI(ed_loc.x + i , st_loc.y + j, sensed_time)
                    if j == 0 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif i == x-1 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i][j-1], dp[i+1][j])
            return dp[0][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y <= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = ed_loc.y - st_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0, x):
                for j in range(0, y):
                    rci = self.get_RCI(st_loc.x + i, st_loc.y + j, sensed_time)
                    if i == 0 and j != 0:
                        dp[i][j] = rci + dp[i][j-1]
                    elif j == 0 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == 0:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i][j-1], dp[i-1][j])
            return dp[x-1][y-1]
        elif st_loc.x <= ed_loc.x and st_loc.y >= ed_loc.y:
            x = ed_loc.x - st_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(0,  x):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(st_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == 0 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != 0:
                        dp[i][j] = rci + dp[i-1][j]
                    elif i == 0 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i-1][j], dp[i][j+1])
            return dp[x-1][0]
        elif st_loc.x >= ed_loc.x and st_loc.y >= ed_loc.y:
            x = st_loc.x - ed_loc.x + 1
            y = st_loc.y - ed_loc.y + 1
            dp = [[0 for j in range(y)] for i in range(x)]
            for i in range(x - 1, -1, -1):
                for j in range(y-1, -1, -1):
                    rci = self.get_RCI(ed_loc.x + i, ed_loc.y + j, sensed_time)
                    if i == x-1 and j != y - 1:
                        dp[i][j] = rci + dp[i][j+1]
                    elif j == y-1 and i != x-1:
                        dp[i][j] = rci + dp[i+1][j]
                    elif i == x-1 and j == y-1:
                        dp[i][j] = rci
                    else:
                        dp[i][j] = rci + max(dp[i+1][j], dp[i][j+1])
            return dp[0][0]
        
    def get_ASV(self, x, y, t) -> float:
        asv = self.cur_asv_map[x][y][t]
        asv = asv if asv >= 0 else 0
        asv = asv if asv <= SPEED_LEVEL else SPEED_LEVEL
        return asv
    
    def get_NOV(self, x, y, t) -> int:
        return self.cur_nov_map[x][y][t]
    
    def get_RCI(self, x, y, t) -> float:
        alpha = 0.4
        beta = 0.6
        if x >= 0 and x <= 50 and y >= 0 and y <= 70:
            mi = np.min(np.array(self.cur_nov_map[x][y]))
            ma = np.max(np.array(self.cur_nov_map[x][y]))
            ma = 1 if ma == 0 else ma
            RCI = alpha * (self.cur_nov_map[x][y][t] - mi) / (ma - mi) + beta * (1 - self.get_ASV(x, y, t) / SPEED_LEVEL)
        else:
            print("error", x, y, t)
            RCI = 0.5
        return RCI
    
    def get_TU(self) -> tuple:
        # TU0, TU1
        TU0 = 0
        TU1 = 0
        for task in self.tasks:
            if task.type == TASK_TYPE[0]:
                TU0 += task.get_waiting_time(MAX_TIME_SLICE)
            if task.type == TASK_TYPE[1]:
                TU1 += task.data
        TU0 = TU0 / self.n_type[0] / 48 if self.n_type[0] != 0 else 0
        TU1 = TU1 / self.n_type[1] if self.n_type[1] != 0 else 0
        
        # TU2
        a = 0
        b = 0
        for task in self.tasks:
            if task.type == TASK_TYPE[2]:
                wt = task.data
                a += wt
                b += wt ** 2
        TU2 = (a ** 2) / (self.n_type[2] * b) if b != 0 else 0
        
        # TU
        TU = -TU0 * (self.n_type[0] / N_TASK) + TU1 * \
            (self.n_type[1] / N_TASK) + TU2 * (self.n_type[2] / N_TASK)
        
        return TU, TU0, TU1, TU2
    
    def get_EP(self) -> float:
        EP = 0
        EP2 = 0
        for vehicle in self.vehicles:
            EP += vehicle.reward
            EP2 += vehicle.reward / vehicle.dc if vehicle.dc != 0 else 0
        return round(EP / N_VEHICLE, 4), round(EP2 / N_VEHICLE, 4)
    
    def get_f(self) -> float:
        TU, _, _, _ = self.get_TU()
        FI, _, _ = self.get_FI()
        _, EP = self.get_EP()
        TC = self.get_coverage()
        
        return 2 * TU + 2 * FI - 0.5 * EP + TC
    
    def get_coverage(self) -> float:
        TC = 0
        for task in self.tasks:
            if task.data > 0:
                TC += 1
        TC /= N_TASK
        return TC
    
    def get_stats(self):
        FI, SF, TF = self.get_FI()
        TU, TU0, TU1, TU2 = self.get_TU()
        EP, EP2 = self.get_EP()
        f = self.get_f()
        TC = self.get_coverage()
        CPC = EP / (TC * 100)
        return [round(FI, 4), round(TF, 4), round(SF, 4), round(TU, 4), round(TU0, 4), round(TU1, 4), round(TU2, 4), round(EP, 4), round(TC, 4), round(CPC, 4), EP2]