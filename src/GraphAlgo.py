import codecs
import random
from asyncio import PriorityQueue
from typing import List, io
import json

import numpy as np

from DiGraph import DiGraph
from GraphAlgoInterface import GraphAlgoInterface
from GraphInterface import GraphInterface


class GraphAlgo(GraphAlgoInterface):
    """This abstract class represents an interface of a graph."""

    def __init__(self, graph:DiGraph)-> None:
        self.nodes = graph.nodes
        self.edges = {(e['src'], e['dest']): e['w'] for e in graph.edges}
        self.graph = DiGraph(self.nodes,self.edges)

    def __init__(self)-> None:
        self.nodes = {}
        self.edges = {}
        self.graph = DiGraph()

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
                if(len(dict["Nodes"][n])==1):
                    tuple = [np.random.uniform(35, 36), np.random.uniform(32, 33)]
                    graph.add_node(id, tuple)
                else:
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
            self.graph=graph

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
        unvisited = list(self.nodes.keys())

        shortest_from_src = {i:float('inf') for i  in unvisited} #dist between src and other nodes
        shortest_from_src[src] = 0 #dist from src to itself is 0

        previous_nodes=[]

        while unvisited:
            current = None
            #let's find the node with the lowest weight value
            for node in unvisited:
                if current == None:
                    current=node
                elif shortest_from_src[node]< shortest_from_src[current]:
                    current = node
            neighbors = self.graph.all_out_edges_of_node(current)
            for i in neighbors:
                m = list(neighbors[i])
                value = shortest_from_src[current] + neighbors[i].get(m[0])
                if value < shortest_from_src[m[0]]:
                    shortest_from_src[m[0]]=value
                    previous_nodes.insert(i,current)
            unvisited.remove(current)

        return previous_nodes,shortest_from_src


        # unvisited = list(self.nodes.keys())
        # distance_from_src = {} #dist between src and other nodes
        # shortest_path = {}
        #
        # for i in unvisited:
        #     unvisited[i]=float('inf')
        # # unvisited[0]=0 #dist from src to itself is 0
        # print(self.graph.v_size())
        # self.graph.add_node(0,2)
        # print(self.graph.v_size())
        # current = self.graph.all_out_edges_of_node(src)
        # # print(self.get_graph())
        # # print(self.graph)
        # # print("unvisited:",unvisited)
        # print("current",current)




        # Dist_from_src = {v: float('inf') for v in range(self.nodes)} #dist between src and other nodes
        # Dist_from_src[src] = 0 #dist from src to itself is 0
        # visited = {i:False for i in range(self.nodes)} #haven't visited and nodes
        #
        # pq = PriorityQueue()
        # # for i in range(len(self.nodes)) pq.put()
        # pq.put((0, src))
        #
        # while not pq.empty():
        #     (dist, current_vertex) = pq.get()
        #     self.visited.append(current_vertex)
        #
        # for neighbor in range(self.nodes):
        #     if self.edges[current_vertex][neighbor] != -1:
        #         distance = self.edges[current_vertex][neighbor]
        #         if neighbor not in self.visited:
        #             old_cost = D[neighbor]
        #             new_cost = D[current_vertex] + distance
        #             if new_cost < old_cost:
        #                 pq.put((new_cost, neighbor))
        #                 Dist[neighbor] = new_cost
        # return D

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        answer = []
        # print(self.dijkstra(id1)[0])
        print(self.dijkstra(id1))
        # i=0
        # while i != id2:
        #     answer.insert(i ,self.dijkstra(id1)[0][i])
        # self.dijkstra(id1)[1][id2]
        return 0.8927487, answer

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