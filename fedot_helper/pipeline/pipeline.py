from enum import Enum

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional

from fedot_helper.estimator._estimator import Estimator


from fedot_helper.estimator.catboost_estimator import CatBoostRegressor
from fedot_helper.task import Task


# TODO tests


class SplitTypesEnum(Enum):
    h = 'h'
    v = 'v'


class Node:
    def __init__(self, estimator: Estimator,
                 parent: Optional[List] = None,
                 child: Optional[List] = None,
                 split_type: SplitTypesEnum = SplitTypesEnum.h,
                 default_model: Optional[Estimator] = None,
                 ):
        self.estimator = estimator
        self.split_type = split_type
        self.default_model = default_model

        self.parent = parent or list()
        self.child = child or list()

        self.fitted = False
        self.predicted = False

        self.achieved_tasks = dict()

    def push(self, tasks: List[Task], parent: 'Node'):
        self.achieved_tasks[parent] = tasks
        if len(self.achieved_tasks) != self.estimator.task_length:
            return
        tasks = list(self.achieved_tasks.values())
        self.achieved_tasks = dict()
        return self.estimator.fit_predict(tasks)


class Pipeline:
    # TODO use NetworkX
    def __init__(self, list_of_nodes: List):
        for n1, n2 in list_of_nodes:
            n1.child.add(n2)
            n2.parent.add(n1)
        self.roots = self._get_roots(n1)

    @classmethod
    def _get_roots(cls, node):
        old_nodes = set()
        nodes = [node]
        nodes_without_child = []
        while nodes:
            node = nodes.pop()
            old_nodes.add(node)
            nodes.extend([x for x in node.child if x not in old_nodes])
            nodes.extend([x for x in node.parent if x not in old_nodes])

            if not node.child:
                nodes_without_child.append(node)
        return nodes_without_child

n1 = Node(CatBoostRegressor())
n2 = Node(CatBoostRegressor())
n3 = Node(CatBoostRegressor())

g = Pipeline([[n1, n2], [n2, n3]])
