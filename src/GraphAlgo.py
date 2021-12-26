from typing import List
import json
import numpy as np
from numpy import double
from DiGraph import DiGraph
from GraphAlgoInterface import GraphAlgoInterface
from GraphInterface import GraphInterface
import random
import matplotlib.pyplot as plt
import numpy as np


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
            print("edgelen:", len(dict_e))
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
                dict_n = {"Nodes": self.nodes}
                print(len(dict_n))
                json.dump(dict_e, indent=2, fp=f, default=lambda a: a.__dict__)
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

    def dijkstra(self, src: int) -> (list, list):
        unvisited = list(self.nodes.keys())

        shortest_from_src = {i:float('inf') for i in unvisited} #dist between src and other nodes
        shortest_from_src[src] = 0 #dist from src to itself is 0

        previous_nodes=[]

        while unvisited:
            current = None
            #let's find the node with the lowest weight value
            for node in unvisited:
                if current == None:
                    current = node
                elif shortest_from_src[node] < shortest_from_src[current]:
                    current = node

            neighbors = self.graph.all_out_edges_of_node(current)

            for i in range(len(neighbors)):
                m = list(neighbors[i])
                value = shortest_from_src[current] + neighbors[i].get(m[0])
                if value < shortest_from_src[m[0]]:
                    shortest_from_src[m[0]]=value
                    previous_nodes.insert(m[0],current)


            unvisited.remove(current)

        return previous_nodes, shortest_from_src

    def is_connected(self):
        return float('inf') not in self.dijkstra(0)[1]

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        answer = []
        node = id2

        list = self.dijkstra(id1)
        print(list)
        print(list[0])

        while node != id1:
            answer.append(node)
            node = list[0][node]

        answer.append(id1)
        result = answer[::-1]

        return list[1][id2], result

    def TSP(self, node_lst: List[int]) -> (List[int], float):
        if not self.is_connected():
            return [],0.0

        copy_cities = [j for j in node_lst]# copy node list
        result = []
        answer = 0

        temp = node_lst[0]
        result.append(copy_cities[0])
        copy_cities.remove(copy_cities[0])

        while len(copy_cities)>=1:
            min = double('inf')
            same = -1
            place = -1
            for i in range(len(copy_cities)):
                open = copy_cities[i]
                dist = self.shortest_path(temp, open)[0]
                if dist < min:
                    min = dist
                    same = open
                    place = i
            list = self.shortest_path(temp,same)[1]
            while len(list)>=1:
                if list[0] not in result:
                    result.append(list[0])
                list.remove(list[0])
            q = copy_cities[place]
            temp=q
            copy_cities.remove(copy_cities[place])
            if len(copy_cities)==1 and same+1 not in result:
                result.append(same+1)

        return result, answer

        """
        Finds the shortest path that visits all the nodes in the list
        :param node_lst: A list of nodes id's
        :return: A list of the nodes id's in the path, and the overall distance
        """



    def centerPoint(self) -> (int, float):

        if not self.is_connected():
            return None, None

        list = []

        for i in range(len(self.nodes)):
            dist = self.dijkstra(i)[1] # list of distances
            # find maximum
            max=0
            for j in range(len(dist)):
                if dist[j]>max:
                    max=dist[j]
            list.insert(i,max)

        min = float('inf')

        for i in range(len(list)):
            if min>list[i]:
                min=list[i]
                node = i

        return node, min

        """
        Finds the node that has the shortest distance to it's farthest node.
        :return: The nodes id, min-maximum distance
        """

    def plot_graph(self) -> None:
        # x_vals = [1,2,3,4]
        # y_vals = [1,4,9,16]
        # plt.plot(x_vals,y_vals,label = "My firs plot :)")
        # plt.xlabel("x axis ")
        # plt.ylabel("y axis ")
        # plt.title("The title of the graph")
        # plt.legend()
        # plt.show()
        #
        # x = np.arange(0,10,0.1)
        # plt.figure(figsize=(40,40))
        # y = np.sin(x)
        # plt.plot(x,y,"D-")
        # plt.plot(x_vals,y_vals,"ro-")
        # plt.show()
        # #
        # #
        # x = [0.15, 0.3, 0.45, 0.6, 0.75]
        # y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
        # n = [58, 651, 393, 203, 123]

        # fig, ax = plt.subplots()
        # ax.scatter(x, y)
        #
        # for i, txt in enumerate(n):
        #     ax.annotate(n[i], (x[i]+0.005, y[i]+0.005)) # arrowprops=dict(arrowstyle="simple")
        #
        # plt.plot(x, y)
        # plt.show()
        #
        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        #
        # z_line = np.linspace(0, 15, 100)
        # x_line = np.cos(z_line)
        # y_line = np.sin(z_line)
        # ax.plot3D(x_line, y_line, z_line, 'gray')
        #
        # z_points = 15 * np.random.random(100)
        # x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
        # y_points = np.sin(z_points) + 0.1 * np.random.randn(100)
        # ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')
        #
        # plt.show()

        x = []
        # for i in range(self.graph.v_size()):
        # x.append(self.nodes.get(i))
        print(type(self.nodes.get(0)))
        # y=[1,7,3,4]
        # plt.plot(x,y,'go-')
        # plt.title("oop oop")
        # plt.xlabel("x")
        # plt.ylabel("y")

        # c=np.arange(0,10,0.01)
        # plt.plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)
        # plt.plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')
        # cos=np.sin(c)
        # plt.plot(c,cos)
        # plt.plot(0, 0, markersize=10, marker='.', color='blue')
        # plt.plot(x, y, markersize=10, marker='.', color='blue')
        # plt.text(x, y, str(i.getkey()), color="red", fontsize=12)
        plt.show()

        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
