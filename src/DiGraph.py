from GraphInterface import GraphInterface
from Node import Node

class DiGraph(GraphInterface):

    def __init__(self)-> None:
        self.nodes = {}
        self.edges = {}
        self.mc = 0

    def __repr__(self) -> str:
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    def v_size(self) -> int:
        return len(self.nodes)

    def e_size(self) -> int:
        return len(self.edges)

    def get_all_v(self) -> dict:
        return self.nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        ans = {}
        for key,value in self.edges.items():
            if value['dest']==id1:
                ans[len(ans)] = {(value['src']):value['w']}
        return ans

    def all_out_edges_of_node(self, id1: int) -> dict:
        ans = {}
        for key, value in self.edges.items():
            if value['src'] == id1:
                ans[len(ans)] = {(value['dest']): value['w']}
        return ans

    def get_mc(self) -> int:
        return self.mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        if id1 in self.nodes.keys() and id2 in self.nodes.keys():
            self.edges[self.e_size()]={'src':id1,'w': weight, 'dest':id2}
            self.mc += 1
            return True
        return False

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        if node_id in self.nodes:
            return False
        else:
            self.nodes[node_id] = Node(node_id, pos)
            self.mc += 1
            return True

    def remove_node(self, node_id: int) -> bool:
        # erase the node itself
        if self.nodes.__contains__(node_id):
            del self.nodes[node_id]

        # erase all connected edges using other functions
            for key, value in list(self.edges.items()):
                if value['src'] == node_id:
                    del self.edges[key]
                elif value['dest'] == node_id:
                    del self.edges[key]
            self.mc += 1
            return True
        return False

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
       if node_id1 in self.nodes and node_id2 in self.nodes:
           for key, value in list(self.edges.items()):
               if value['src'] == node_id1 and value['dest'] == node_id2:
                   del self.edges[key]
           self.mc += 1
           return True
       return False

    def get_edge(self, id1, id2):
        return self.edges.get(id1, id2)