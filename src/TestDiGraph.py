import unittest
from operator import pos

from src.DiGraph import DiGraph


class MyTestCase(unittest.TestCase):
    def test_v_size(self):
        g = DiGraph()
        for n in range(4):
            g.add_node(n)
        self.assertEqual(g.v_size(), 4)

    def test_e_size(self):
        g = DiGraph()
        for n in range(4):
            g.add_node(n)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 0, 1.1)
        g.add_edge(1, 2, 1.3)
        g.add_edge(2, 3, 1.1)
        g.add_edge(1, 3, 1.9)
        self.assertEqual(g.e_size(), 5)

    def test_get_all_v(self):
        g = DiGraph()
        for n in range(4):
            g.add_node(n)
        self.assertEquals(g.get_all_v(), g.nodes)

    def test_get_mc(self):
        g1 = DiGraph()
        g1.add_node(0, {2,3})
       # g.add_edge(0, 1, 1)
        print(g1.v_size())
        self.assertEquals(g1.get_mc(),1)

if __name__ == '__main__':
    unittest.main()