from __future__ import annotations
from env.location import Location


class Vehicle:

    def __init__(self, id=0, traj=[], type=0) -> None:
        self.id = id
        self.loc = Location(traj[0][1], traj[0][2])
        self.traj = traj # 真实轨迹数据集, [时间, x, y, 速度, 载客]
        self.type = type # 0为正常用户, 1为恶意用户

        # 统计信息
        self.logs = [] # 记录每次的 [任务, 感知时间, 奖励, 距离]
        self.reward = 0
        self.dc = 0
    
    def get_info(self, time_slice: int) -> list:
        return self.loc[time_slice]

    def is_free(self, time_slice: int) -> bool:
        if time_slice >= len(self.traj) or self.traj[time_slice][4] == 1:
            return False
        else:
            return True

    def sense(self, task_id: int, sensed_time: int, reward: int, dc: int) -> bool:
        self.logs.append([task_id, sensed_time, reward, dc])
        self.reward += reward
        self.dc += dc
    
    def de_sense(self, task_id: int, sensed_time: int, reward: int, dc: int) -> bool:
        if [task_id, sensed_time, reward, dc] in self.logs:
            self.logs.remove([task_id, sensed_time, reward, dc])
            self.reward -= reward
            self.dc -= dc
        else:
            print("error vehicle de_sense")

    def __repr__(self) -> str:
        return "Vehicle_{}:Loc({})".format(self.id, self.loc)
