from __future__ import annotations
from env.location import Location


class Task:

    def __init__(self, id: int, loc=Location(), arrival_time=0, type="discrete") -> None:
        self.id = id
        self.loc = loc
        self.arrival_time = arrival_time
        self.type = type  # discrete, continuous, periodic
        self.data = 0
        self.done = False
        self.logs = []  # 记录每次的 [车辆, 感知时间]
    
    def is_free(self, time_slice: int) -> bool:
        if self.done == False and self.arrival_time <= time_slice:
            return True
        else:
            return False
    
    def get_waiting_time(self, time_slice: int) -> int:
        if len(self.logs) > 0:
            return self.logs[0][1] - self.arrival_time
        else:
            return time_slice - self.arrival_time
    
    def sense(self, vehicle_id: int, sensed_time: int) -> bool:
        if self.done == False and sensed_time >= self.arrival_time:
            self.logs.append([vehicle_id, sensed_time])
            self.data += 1
            if self.type != "periodic":
                self.done = True
            return True
        else:
            return False
    
    def de_sense(self, vehicle_id: int, sensed_time: int) -> bool:
        if [vehicle_id, sensed_time] in self.logs:
            self.logs.remove([vehicle_id, sensed_time])
            self.data -= 1
            self.done = False

    def __repr__(self) -> str:
        return "Task_{}:Loc({}),AT({}),Data({}),Done({})".format(self.id, self.loc, self.arrival_time, self.data, self.done)
