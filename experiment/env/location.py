from __future__ import annotations
import math


class Location:
    
    def __init__(self, x=0, y=0) -> None:
        # 位置坐标(x.y)
        self.x = x
        self.y = y

    def __sub__(self, __o: object) -> int:
        return math.sqrt((self.x - __o.x) ** 2 + (self.y - __o.y) ** 2)

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y

    def __str__(self) -> str:
        return "{},{}".format(self.x, self.y)
