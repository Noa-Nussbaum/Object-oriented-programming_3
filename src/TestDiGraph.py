import unittest
from src.DiGraph import DiGraph


class MyTestCase(unittest.TestCase):
    g = DiGraph()
    for n in range(4):
        g.add_node(n)
    g.add_edge(0, 1, 1)
    g.add_edge(1, 0, 1.1)
    g.add_edge(1, 2, 1.3)
    g.add_edge(2, 3, 1.1)
    g.add_edge(1, 3, 1.9)
    def test_v_size(self):
        self.assertEqual(self.g.v_size(), 4)

    def test_e_size(self):
        self.assertEqual(self.g.e_size(), 5)

    def test_get_all_v(self):
        self.assertEquals(self.g.get_all_v(), self.g.nodes)

    def test_get_mc(self):
        self.assertEquals(self.g.get_mc(),9)

    def test_add_node(self):
        self.g.add_node(4)
        self.g.add_node(5)
        self.g.add_node(6)
        self.assertEqual(self.g.v_size() ,7)

    def test_add_edge(self):
        self.g.add_node(5)
        self.g.add_node(6)
        self.g.add_node(7)
        self.g.add_edge(5,6,4)
        self.g.add_edge(5,3,2)
        self.g.add_edge(1,6,3)
        self.assertEqual(self.g.e_size(), 8)

    def test_remove_node(self):
        self.g.remove_node(2)
        self.g.remove_node(3)
        self.assertEqual(self.g.v_size(), 2)

    def test_remove_edge(self):
        self.g.remove_edge(2,3)
        self.g.remove_edge(1,0)
        self.assertEqual(self.g.e_size(), 3)

    def test_all_in_edges_of_node(self):
        self.g.add_edge(2, 1, 4)
        dict_in={(2,4),(0,1)} #src, weihgt
        self.assertEquals(dict_in, self.g.all_in_edges_of_node(1))

    def test_all_out_edges_of_node(self):
        dict_out={(0,1.1),(2, 1.3),(3, 1.9)}   #dest, weihgt
        self.assertEquals(dict_out, self.g.all_out_edges_of_node(1))

if __name__ == '__main__':
    unittest.main()