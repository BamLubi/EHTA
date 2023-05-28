from env.vcs import VCS
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

vcs = VCS(n_vehicle, malicious=malicious)


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
