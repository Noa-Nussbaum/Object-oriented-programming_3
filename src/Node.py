import json
import matplotlib.pyplot as plt
import numpy as np


class Node:
    # count=0
    def __init__(self, id, pos: dict) -> None:
        self.pos = pos
        self.id = id

    def __repr__(self) -> str:
        return f"id:{self.id} pos:{self.pos}"




