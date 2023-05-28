from __future__ import annotations


class Location:
    def __init__(self, x=0, y=0) -> None:
        self.x = x
        self.y = y

    def __sub__(self, __o: object) -> int:
        return abs(self.x - __o.x) + abs(self.y - __o.y)

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y

    def __str__(self) -> str:
        return "{},{}".format(self.x, self.y)
