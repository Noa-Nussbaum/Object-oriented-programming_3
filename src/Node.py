
class Node:

    def __init__(self, id, pos: tuple=None) -> None:
        self.pos = pos
        self.id = id

    def __repr__(self) -> str:
        return f"id:{self.id} pos:{self.pos}"

    def getId(self):
        return self.id

    def getPos(self):
        return self.pos

    def setId(self, i: int):
        self.id = i

    def setPos(self,pos: tuple):
        self.pos = pos

    def __repr__(self) -> str:
        return f"id:{self.id} pos:{self.pos}"










