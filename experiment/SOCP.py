from env.vcs import VCS, sigmoid
import numpy as np

np.random.seed(2023)

n_vehicle = 10
# n_vehicle = 20
# n_vehicle = 30
# n_vehicle = 40
# n_vehicle = 50
malicious = 0
# malicious = 0.05
# malicious = 0.1
# malicious = 0.15
# malicious = 0.2

vcs = VCS(n_vehicle, malicious)


def get_SOCP_target(vehicle_id, task_id, sensed_time) -> float:
    dc = vcs.vehicles[vehicle_id-1].loc - vcs.tasks[task_id-1].loc
    vcs.pre_sense(vehicle_id, task_id, sensed_time)
    new_TC = vcs.get_coverage()
    vcs.pre_desense(vehicle_id, task_id, sensed_time)
    cost_1 = (1 - new_TC) * 0.25
    cost_2 = new_TC * 0.3
    ans = cost_1 + cost_2 - dc / 4000
    ans = - (new_TC * 0.9 + dc / 4000)
    return ans


def ALG_SOCP(vehicles, tasks, time_slice) -> list[int]:
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
                r = get_SOCP_target(vehicles[i], tasks[j], time_slice)
                if r > max_reward:
                    max_reward = r
                    sel_task = j

        visit[sel_task] = 1
        decision[i] = tasks[sel_task]

        allocated_tasks_cnt += 1
        if allocated_tasks_cnt >= len(tasks):
            break
    return decision


vcs.run(ALG_SOCP)

vcs.get_stats()
