import unittest
from src import GraphAlgo
from src.GraphAlgo import GraphAlgo


class MyTestCase(unittest.TestCase):

    def test_load_from_json_1000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\1000Nodes.json"
        g_algo.load_from_json(file)
        self.assertEqual(g_algo.load_from_json(file), True)

    def test_save_to_json_1000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\1000Nodes.json"
        g_algo.load_from_json(file)
        self.assertEqual(g_algo.save_to_json(file+'saved'), True)


    def test_shortest_path_1000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\1000Nodes.json"
        g_algo.load_from_json(file)
        dist, path= g_algo.shortest_path(5, 36)
        self.assertEqual(g_algo.shortest_path(5,36), (dist, path))

    def test_centerPoint_1000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\1000Nodes.json"
        g_algo.load_from_json(file)
        node, min = 362, 1185.9594924690523
        self.assertEqual(g_algo.centerPoint(), (node, min))

    def test_TSP_1000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\1000Nodes.json"
        g_algo.load_from_json(file)
        result, answer =([23, 480, 328, 98, 780, 186, 544, 896, 166, 448, 374, 497, 922, 506, 220, 45, 46, 123, 692, 674, 870, 683, 940,
          2, 590, 34], 17994399.764941745)
        self.assertEqual(g_algo.TSP([23,34,45,506,780]), (result, answer))




    def test_load_from_json_10000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\10000Nodes.json"
        g_algo.load_from_json(file)
        self.assertEqual(g_algo.load_from_json(file), True)

    def test_save_to_json_10000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\10000Nodes.json"
        g_algo.load_from_json(file)
        self.assertEqual(g_algo.save_to_json(file+'saved'), True)

    def test_shortest_path_10000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\10000Nodes.json"
        g_algo.load_from_json(file)
        dist, path= (1317.7359901137258, [20, 2128, 8677, 703, 6644, 1141, 50])
        self.assertEqual(g_algo.shortest_path(20,50), (dist, path))

    def test_centerPoint_10000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\10000Nodes.json"
        g_algo.load_from_json(file)
        node, min = g_algo.centerPoint()
        self.assertEqual(g_algo.centerPoint(), (node, min))

    def test_TSP_10000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\10000Nodes.json"
        g_algo.load_from_json(file)
        result, answer= ([10, 1008, 7950, 9371, 22, 2418, 6142, 334, 335, 5275, 6476, 8035, 5239, 911, 6143, 1823, 2776, 2],
         89826349.26090957)
        self.assertEqual(g_algo.TSP([10, 2, 334]), (result, answer))

    def test_load_from_json_100000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\100000.json"
        g_algo.load_from_json(file)
        self.assertEqual(g_algo.load_from_json(file), True)

    def test_save_to_json_100000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\100000.json"
        g_algo.load_from_json(file)
        self.assertEqual(g_algo.save_to_json(file+'saved'), True)

    def test_shortest_path_100000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\100000.json"
        g_algo.load_from_json(file)
        dist, path = g_algo.shortest_path(5, 36)
        print(dist, path)
        self.assertEqual(g_algo.shortest_path(5, 36), (dist, path))

    def test_centerPoint_100000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\100000.json"
        g_algo.load_from_json(file)
        node, min = 362, 1185.9594924690523
        self.assertEqual(g_algo.centerPoint(), (node, min))

    def test_TSP_100000(self):
        g_algo = GraphAlgo()
        file = "C:\\Users\\User\\PycharmProjects\\Ex3\\JSON files\\100000.json"
        g_algo.load_from_json(file)


if __name__ == '__main__':
    unittest.main()