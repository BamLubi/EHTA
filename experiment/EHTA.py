import math
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from env.vcs import VCS, sigmoid


def RUN(n_vehicle, malicious, seed, task_type, vehicle_set, fix_time):
    
    np.random.seed(seed)
    
    vcs = VCS(n_vehicle, malicious=malicious, alg_flg=1, malicious_flg=0, seed=seed, task_type=task_type, vehicle_set=vehicle_set, fix_time=fix_time)

    f_list = []

    def metropolis(e, new_e, t):
        if new_e < e:
            return True
        else:
            p = math.exp((new_e - e) / t)
            return True if random.random() < p else False

    def gen_x(vehicles, tasks):
        visit = [0] * len(tasks)
        decision = [0] * len(vehicles)
        allocated_tasks_cnt = 0
        for i in range(0, len(vehicles)):
            task_id = np.random.randint(0, len(tasks), 1)[0]
            while visit[task_id] == 1:
                task_id = np.random.randint(0, len(tasks), 1)[0]
            visit[task_id] = 1
            decision[i] = tasks[task_id]
            
            allocated_tasks_cnt += 1
            if allocated_tasks_cnt >= len(tasks):
                break
        return decision

    def search(all_x, all_e):
        max_e = -1000
        max_idx = -1
        for i in range(0, len(all_e)):
            if all_e[i] > max_e:
                max_e = all_e[i]
                max_idx = i
        return all_x[max_idx] if max_idx != -1 else [], max_e

    def ALG_EHTA(vehicles, tasks, time_slice) -> list[int]:
        if len(tasks) == 0 or len(vehicles) == 0:
            return []
        # 模拟退火算法求解
        init = gen_x(vehicles, tasks)
        all_x = []
        all_e = []
        t = 2000
        t_final = 1e-2
        alpha = 0.99
        while t > t_final:
            x, e = search(all_x, all_e)
            for i in range(10):
                new_x = gen_x(vehicles, tasks)
                for i in range(0, len(new_x)):
                    vcs.pre_sense(vehicles[i], new_x[i], time_slice)
                new_e = vcs.get_f()
                for i in range(0, len(new_x)):
                    vcs.pre_desense(vehicles[i], new_x[i], time_slice)
                if metropolis(e, new_e, t):
                    x = new_x
                    e = new_e
                    all_x.append(x)
                    all_e.append(e)
                    f_list.append(e)
            t = alpha * t
        x, e = search(all_x, all_e)
        return x

    vcs.run(ALG_EHTA)

    return vcs.get_stats()

if __name__ == '__main__':
    RUN(50, 0, 2023, 'uniform', "vechiles_1", 0)