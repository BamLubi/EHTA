import numpy as np
from env.vcs import VCS, sigmoid

vcs = VCS()

def get_FCSMP_target(vehicle_id, task_id, sensed_time) -> float:
    
    vcs.pre_sense(vehicle_id, task_id, sensed_time)
    new_TC = vcs.get_coverage()
    vcs.pre_desense(vehicle_id, task_id, sensed_time)
    
    return new_TC * 1000

def ALG_FCSMP(vehicles, tasks, time_slice) -> list[int]:
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
                r = get_FCSMP_target(vehicles[i], tasks[j], time_slice)
                if r > max_reward:
                    max_reward = r
                    sel_task = j
        
        visit[sel_task] = 1
        decision[i] = tasks[sel_task]
        
        allocated_tasks_cnt += 1
        if allocated_tasks_cnt >= len(tasks):
            break
    return decision


vcs.run(ALG_FCSMP)

vcs.get_stats()
    