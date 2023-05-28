from __future__ import annotations

from env.location import Location


class Vehicle:

    def __init__(self, id=0, traj=[], type=0) -> None:
        self.id = id
        self.loc = Location(traj[0][1], traj[0][2])
        self.traj = traj
        self.type = type

        self.logs = []
        self.reward = 0
    
    def get_info(self, time_slice: int) -> list:
        return self.loc[time_slice]

    def is_free(self, time_slice: int) -> bool:
        if time_slice >= len(self.traj) or self.traj[time_slice][4] == 1:
            return False
        else:
            return True

    def sense(self, task_id: int, sensed_time: int, reward: int) -> bool:
        self.logs.append([task_id, sensed_time, reward])
        self.reward += reward
    
    def de_sense(self, task_id: int, sensed_time: int, reward: int) -> bool:
        if [task_id, sensed_time, reward] in self.logs:
            self.logs.remove([task_id, sensed_time, reward])
            self.reward -= reward

    def __repr__(self) -> str:
        return "Vehicle_{}:Loc({})".format(self.id, self.loc)
