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
SPEED_LEVEL = 70
TASK_TYPE = ["discrete", "continuous", "periodic"]
COPY_VEHICLE = False

N_VEHICLE = 10
N_TASK = 500

np.random.seed(2023)

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

def copy_vehicle() -> None:
    path = "\\Data\\BlockData\\30\\20150419\\"
    new_path = "\\Experiment\\data\\vechiles\\" + str(N_VEHICLE) + "\\"
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

class VCS(object):
    def __init__(self, n_vehicle=10, malicious=0.2) -> None:
        global N_VEHICLE 
        N_VEHICLE = n_vehicle
        
        self.malicious = malicious
        self.vehicles = self.__gen_vehicle()
        self.n_type = [0, 0, 0]
        self.tasks = self.__gen_task()
        self.cur_nov_map = np.load("\\Experiment\\npy\\20150419_nov.npy")
        self.cur_asv_map = np.load("\\Experiment\\npy\\20150419_asv.npy")

    def __gen_vehicle(self) -> list[Vehicle]:
        if COPY_VEHICLE == True:
            copy_vehicle()
        vehicles = []
        path = "\\Experiment\\data\\vechiles\\" + str(N_VEHICLE) + "\\"
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
                vehicles.append(Vehicle(idx, traj=traj, type=1))
            else:
                vehicles.append(Vehicle(idx, traj=traj))
            idx += 1
        return vehicles

    def __gen_task(self) -> list[Task]:
        tasks = []
        x_list = np.random.randint(0, CITY_M, N_TASK)
        y_list = np.random.randint(0, CITY_N, N_TASK)
        at_list = np.random.randint(MIN_TIME_SLICE, MAX_TIME_SLICE, N_TASK)
        for idx in range(0, N_TASK):
            type = np.random.randint(0, 3, 1)[0]
            self.n_type[type] += 1
            tasks.append(
                Task(idx + 1, Location(x_list[idx], y_list[idx]), at_list[idx], TASK_TYPE[type]))
        return tasks

    @staticmethod
    def __gen_map() -> list[list[int]]:
        return [[0 for j in range(CITY_N)] for i in range(CITY_M)]
    
    def run(self, allocation_alg: function) -> None:
        for time_slice in tqdm(range(MIN_TIME_SLICE, MAX_TIME_SLICE)):
            tasks = self.get_avl_tasks(time_slice)
            vehicles = self.get_avl_vehicles(time_slice)
            decision = allocation_alg(vehicles, tasks, time_slice)
            for i in range(0, len(decision)):
                if decision[i] == 0:
                    continue
                r = self.get_reward(vehicles[i], decision[i], time_slice)
                # r = self.get_reward_normal(vehicles[i], decision[i], time_slice)
                if self.vehicles[vehicles[i] - 1].type == 1:
                    continue
                self.vehicles[vehicles[i]-1].sense(decision[i], time_slice, r)
                self.tasks[decision[i]-1].sense(vehicles[i], time_slice)
    
    def pre_sense(self, vehicle_id, task_id, sensed_time) -> float:
        self.tasks[task_id-1].sense(vehicle_id, sensed_time)
    
    def pre_desense(self, vehicle_id, task_id, sensed_time) -> float:
        self.tasks[task_id-1].de_sense(vehicle_id, sensed_time)
    
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
    
    def get_reward(self, vehicle_id, task_id, sensed_time) -> float:
        # r = EC
        # EC = e^{RCI}(TC + DC)
        
        ## DC
        DC = self.vehicles[vehicle_id-1].loc - self.tasks[task_id-1].loc
        if DC == 0:
            DC = 1
        
        ## RCI
        min_rci = self.get_min_RCI(vehicle_id, task_id, sensed_time) / DC
        max_rci = self.get_max_RCI(vehicle_id, task_id, sensed_time) / DC
        
        ## EC
        TC = self.get_TC(vehicle_id, task_id, sensed_time)
        EC = math.exp(min_rci - 0.5) * (TC + DC)
        MECT = math.exp(max_rci - 0.5) * (TC + DC)
        
        ## r
        r = 0
        if EC > MECT:
            r = 0
        else:
            r = EC
        return r
    
    def get_reward_normal(self, vehicle_id, task_id, sensed_time) -> float:
        # r = TC + DC
        DC = self.vehicles[vehicle_id-1].loc - self.tasks[task_id-1].loc
        if DC == 0:
            DC = 1

        TC = self.get_TC(vehicle_id, task_id, sensed_time)
        r = TC + DC
        if self.vehicles[vehicle_id - 1].type == 1:
            r *= 10
        
        return r
    
    def get_TC(self, vehicle_id, task_id, sensed_time) -> tuple:
        DC = self.vehicles[vehicle_id-1].loc - self.tasks[task_id-1].loc
        if DC == 0:
            DC = 1
        
        loc = self.vehicles[vehicle_id - 1].loc
        v = self.cur_asv_map[loc.x][loc.y][sensed_time] if self.vehicles[vehicle_id-1].traj[sensed_time][3] != 0 else self.vehicles[vehicle_id-1].traj[sensed_time][3]
        v = SPEED_LEVEL / 3 if v == 0 else v
        TC = 2 * DC * math.sqrt(2) / v
        TC = 1 if TC <= 0 else TC
        
        return TC

    def get_FI(self) -> tuple:
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
        
        ## dp 求解最小值
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
        
        ## dp 求解最小值
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
        
    def get_RCI(self, x, y, t) -> float:
        alpha = 0.4
        beta = 0.6
        if x >= 0 and x <= 50 and y >= 0 and y <= 70:
            mi = np.min(np.array(self.cur_nov_map[x][y]))
            ma = np.max(np.array(self.cur_nov_map[x][y]))
            ma = 1 if ma == 0 else ma
            RCI = alpha * (self.cur_nov_map[x][y][t] - mi) / (ma - mi) + beta * (1 - self.cur_asv_map[x][y][t] / SPEED_LEVEL)
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
        TU = -TU0 * (self.n_type[0] / N_TASK) + TU1 * \
            (self.n_type[1] / N_TASK) + TU2 * (self.n_type[2] / N_TASK)
        
        return TU, TU0, TU1, TU2, TU
    
    def get_EP(self) -> float:
        EP = 0
        for vehicle in self.vehicles:
            EP += vehicle.reward
        return EP
    
    def get_f(self) -> float:
        _, _, _, _, TU = self.get_TU()
        EP = self.get_EP() / N_VEHICLE / 1000
        TC = self.get_coverage()
        FI, _, _ = self.get_FI()
        return TU + FI + 10 * TC - 100 * EP
    
    def get_coverage(self) -> float:
        TC = 0
        for task in self.tasks:
            if task.data > 0:
                TC += 1
        TC /= N_TASK
        return TC
    
    def get_stats(self):
        FI, SF, TF = self.get_FI()
        _TU, TU0, TU1, TU2, TU = self.get_TU()
        EP = self.get_EP()
        f = self.get_f()
        TC = self.get_coverage()
        CPC = EP / (TC * 100)
      
        print("Vehicles:{}, Tasks:{}, City:({},{}), Malicious:{}".format(N_VEHICLE, N_TASK, CITY_M, CITY_N, self.malicious))
        print([round(FI, 2), round(TF, 2), round(SF, 2)], [round(TU0,2), round(TU1,2), round(TU2,2), round(TU, 2)], [round(EP, 2), round(TC, 2), round(CPC, 2)])
