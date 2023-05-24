import numpy as np
from env.vcs import VCS

vcs = VCS()

def ALG_random(vehicles, tasks, time_slice) -> list[int]:
    if len(tasks) == 0 or len(vehicles) == 0:
        return []
    
    visit = [0] * len(tasks)
    decision = [0] * len(vehicles)
    allocated_tasks_cnt = 0
    for i in range(0, len(vehicles)):
        task_id = np.random.randint(0, len(tasks), 1)[0]
        while visit[task_id] == 1 or vcs.can_sense(vehicles[i], tasks[task_id], time_slice) == False:
            task_id = np.random.randint(0, len(tasks), 1)[0]
        visit[task_id] = 1
        decision[i] = tasks[task_id]
        
        allocated_tasks_cnt += 1
        if allocated_tasks_cnt >= len(tasks):
            break
    return decision

vcs.run(ALG_random)

vcs.get_stats()