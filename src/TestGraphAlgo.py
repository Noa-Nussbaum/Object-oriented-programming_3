import unittest
from src import GraphAlgo
from src.GraphAlgo import GraphAlgo

g_algo = GraphAlgo()
file = "C:/Users/User/PycharmProjects/Ex3/JSON files/A3.json"

class MyTestCase(unittest.TestCase):
    def test_get_graph(self):
        self.assertEqual(g_algo.graph, g_algo.get_graph())


    def test_load_from_json(self):
        self.assertEqual(g_algo.load_from_json(file), True)

    def test_save_to_json(self):
        self.assertEqual(g_algo.save_to_json('saved_'+"C:/Users/User/PycharmProjects/Ex3/JSON files/A3.json"),True)

    def test_shortest_path(self):
        g_algo = GraphAlgo()
        file = "C:/Users/User/PycharmProjects/Ex3/JSON files/A3.json"
        g_algo.load_from_json(file)
        dist, path = g_algo.shortest_path(4,6)
        self.assertEqual(g_algo.shortest_path(4,6),(dist, path))

    def test_centerPoint(self):
        g_algo = GraphAlgo()
        file = "C:/Users/User/PycharmProjects/Ex3/JSON files/A3.json"
        g_algo.load_from_json(file)
        node, min= 2, 8.182236568942237
        self.assertEqual(g_algo.centerPoint(), (node, min))

    def test_TSP(self):
        g_algo = GraphAlgo()
        file = "C:/Users/User/PycharmProjects/Ex3/JSON files/A3.json"
        g_algo.load_from_json(file)
        result, answer =([0, 21, 22, 23, 24, 25, 26, 8, 7, 44, 43, 42, 41, 40, 39, 17, 14, 15, 38, 37, 36, 35, 34, 33, 32, 2, 3, 31, 30,
          13, 12, 11, 20, 19, 18, 10, 9, 1, 16, 6, 5, 28, 4, 29, 48, 47, 46, 45, 27], 8459.737919535499)

        self.assertEqual(g_algo.TSP(g_algo.nodes), (result, answer))

if __name__ == '__main__':
    unittest.main()