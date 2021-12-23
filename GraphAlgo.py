import codecs
from asyncio import PriorityQueue
from typing import List, io
import json
from DiGraph import DiGraph
from GraphAlgoInterface import GraphAlgoInterface
from GraphInterface import GraphInterface


class GraphAlgo(GraphAlgoInterface):
    """This abstract class represents an interface of a graph."""

    def __init__(self, graph:DiGraph):
        self.nodes = graph.nodes
        self.edges = {(e['src'], e['dest']): e['w'] for e in graph.edges}
        self.edges = graph.edges


    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def __repr__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"

    def get_graph(self) -> GraphInterface:
        return self.graph

    def load_from_json(self, file_name: str) -> bool:
        try:
            graph = DiGraph()
            with open(file_name, "r") as f:
                dict = json.load(f)
            for n in range(len(dict["Nodes"])):
                id = dict["Nodes"][n]["id"]
                pos = dict["Nodes"][n]["pos"]
                tuple = pos.split(',')
                graph.add_node(id, tuple)
            for e in range(len(dict["Edges"])):
                src = dict["Edges"][e]["src"]
                dest = dict["Edges"][e]["dest"]
                w = dict["Edges"][e]["w"]
                graph.add_edge(src,dest,w)
            self.edges=graph.edges
            self.nodes=graph.nodes

            return True
        except Exception:
            return False



    def save_to_json(self, file_name: str) -> bool:
        try:
            dict_e = {"Edges": self.edges}
            print("edgelen:",len(dict_e))
            # dict_n = {"Nodes": self.nodes}
            # dict_n={}
            # print(self.nodes)
            # print("yeah", self.nodes[1]["id"])
            # for i in self.nodes:
            #     dict_n.append({"pos": self.nodes[i], "id": i})
            print(len(self.nodes))
            print(self.nodes[3])
            with open(file_name, 'w') as f:
                # default=lambda a: a.__dict__
                dict_n= {"Nodes":self.nodes}
                print(len(dict_n))
                json.dump(dict_e,indent=2, fp=f,default=lambda a: a.__dict__)
                # json.dump(dict_n,indent=2, fp=f,default=lambda a: a.__dict__)

                # for i in range(len(self.nodes)):
                #     g = self.nodes[i]
                #     json.dump(g, indent=2, fp=f)
                    # json.dump(self.nodes[i], indent=2, fp=f)
                #     g=self.nodes.get(i)
                #     print(g)
                return True
        except Exception:
            return False

    def dijkstra(self, src) -> (list, list):
        D = {v: float('inf') for v in range(self.nodes)}
        D[src] = 0
        visited = {i:False for i in range(self.nodes)}

        pq = PriorityQueue()
        # for i in range(len(self.nodes)) pq.put()
        pq.put((0, src))

        while not pq.empty():
            (dist, current_vertex) = pq.get()
            self.visited.append(current_vertex)

        for neighbor in range(self.nodes):
            if self.edges[current_vertex][neighbor] != -1:
                distance = self.edges[current_vertex][neighbor]
                if neighbor not in self.visited:
                    old_cost = D[neighbor]
                    new_cost = D[current_vertex] + distance
                    if new_cost < old_cost:
                        pq.put((new_cost, neighbor))
                        Dist[neighbor] = new_cost
        return D

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        raise NotImplementedError

    def TSP(self, node_lst: List[int]) -> (List[int], float):
        """
        Finds the shortest path that visits all the nodes in the list
        :param node_lst: A list of nodes id's
        :return: A list of the nodes id's in the path, and the overall distance
        """
        raise NotImplementedError

    def centerPoint(self) -> (int, float):
        """
        Finds the node that has the shortest distance to it's farthest node.
        :return: The nodes id, min-maximum distance
        """

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        raise NotImplementedError