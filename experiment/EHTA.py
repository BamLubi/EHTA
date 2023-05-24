import math
import random

import numpy as np
from env.vcs import VCS, sigmoid

vcs = VCS()

def metropolis(e, new_e, t):
    if new_e >= e:
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
    all_x = []
    all_e = []
    t = 2000
    t_final = 10
    alpha = 0.98
    while t > t_final:
        x, e = search(all_x, all_e)
        for i in range(100):
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
        t = alpha * t
    x, e = search(all_x, all_e)
    return x


vcs.run(ALG_EHTA)

vcs.get_stats()
