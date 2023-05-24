import numpy as np
from env.vcs import VCS, sigmoid

vcs = VCS()


def get_OCA_target(vehicle_id, task_id, sensed_time) -> float:
    vcs.pre_sense(vehicle_id, task_id, sensed_time)
    fi, sf, tf = vcs.get_FI()
    vcs.pre_desense(vehicle_id, task_id, sensed_time)

    return sf


def ALG_OCA(vehicles, tasks, time_slice) -> list[int]:
    if len(tasks) == 0 or len(vehicles) == 0:
        return []

    visit = [0] * len(tasks)
    decision = [0] * len(vehicles)
    allocated_tasks_cnt = 0
    for i in range(0, len(vehicles)):
        max_reward = -1000
        sel_task = -1
        for j in range(0, len(tasks)):
            if visit[j] == 0 and vcs.can_sense(vehicles[i], tasks[j], time_slice) == True:
                r = get_OCA_target(vehicles[i], tasks[j], time_slice)
                if r > max_reward:
                    max_reward = r
                    sel_task = j

        visit[sel_task] = 1
        decision[i] = tasks[sel_task]

        allocated_tasks_cnt += 1
        if allocated_tasks_cnt >= len(tasks):
            break
    return decision


vcs.run(ALG_OCA)

vcs.get_stats()
