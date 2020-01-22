import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
import os
from collections import defaultdict, deque, Counter
from itertools import combinations
from matplotlib.animation import FuncAnimation

class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state == self.goal
    def step_cost(self, s, a, s1): return 1
    def h(self, node):               return 0
    
    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)


class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.

def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.step_cost(s, action, s1)
        yield Node(s1, node, action, cost)

def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]

def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]

def straight_line_distance(A, B):
    "Straight-line distance between two points."
    return sum(abs(a - b)**2 for (a, b) in zip(A, B)) ** 0.5

FIFOQueue = deque

LIFOQueue = list # Stack


class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)

    def add(self, item):
        """Add item to the queue."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]

    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)


class GridProblem(Problem):
    """Finding a path on a 2D grid with obstacles. Obstacles are (x, y) cells."""

    def __init__(self, initial=(0, 0), goal=(9, 9), obstacles=(), **kwds):
        Problem.__init__(self, initial=initial, goal=goal,
                         obstacles=set(obstacles) - {initial, goal}, **kwds)

    directions = [(-1, -1), (0, -1), (1, -1),
                  (-1, 0),           (1,  0),
                  (-1, +1), (0, +1), (1, +1)]
    
    def step_cost(self, s, action, s1): return straight_line_distance(s, s1)
    
    def h(self, node): return straight_line_distance(node.state, self.goal)

    def result(self, state, action): 
        "Both states and actions are represented by (x, y) pairs."
        return action if action not in self.obstacles else state
    
    def draw_walls(self):
        self.obstacles |= {(i, -2) for i in range(-2, self.width+4)}
        self.obstacles |= {(i, self.height+4) for i in range(-2, self.width+4)}
        self.obstacles |= {(-2, j) for j in range(-2, self.height+5)}
        self.obstacles |= {(self.width+4, j) for j in range(-2, self.height+5)}

    def actions(self, state):
        """You can move one cell in any of `directions` to a non-obstacle cell."""
        x, y = state
        return {(x + dx, y + dy) for (dx, dy) in self.directions} - self.obstacles


def random_lines(X=range(0, 10), Y=range(10), N=20, lengths=range(1, 3)):
    """The set of cells in N random lines of the given lengths."""
    result = set()
    for _ in range(N):
        x, y = random.choice(X), random.choice(Y)
        dx, dy = random.choice(((0, 1), (1, 0)))
        result |= line(x, y, dx, dy, random.choice(lengths))
    return result

def line(x, y, dx, dy, length):
    """A line of `length` cells starting at (x, y) and going in (dx, dy) direction."""
    return {(x + i * dx, y + i * dy) for i in range(length)}

def transpose(matrix): return list(zip(*matrix))


class AnimateProblem(GridProblem):
    def __init__(self, solver, weight=1.4, cell_weights=None, **kwargs):
        """Animate the Grid Problem"""
        super().__init__(**kwargs)
        self.weight = weight
        # We may change the cell_weights in case of Uniform Cost search
        self.cell_weights = cell_weights
        if self.cell_weights is None:
            self.cell_weights = np.ones((self.width+5, self.height+5), dtype=np.int16)
        # Define all the allowed solvers and their f-value function.
        # TODO: Bidirectional Search
        self.SOLVERS = {'astar': (lambda n: n.path_cost + self.h(n)),
                        'wastar': (lambda n: n.path_cost + self.weight*self.h(n)),
                        'bfs': (lambda n: len(n)),
                        'dfs': (lambda n: -len(n)),
                        'ucs': (lambda n: n.path_cost),
                        'bestfs': (lambda n: self.h(n))
                       }
        self.solver_f = self.SOLVERS[solver] # Assign the solver's f-value function
        self.solver = solver
        self.__initial_node = Node(self.initial)
        # Dictionary of reach nodes. Simlar to `explored` set.
        self.reached = {self.initial: self.__initial_node}
        # Frontier of nodes to be explored!
        self.frontier = PriorityQueue([self.__initial_node], key=self.solver_f)
        # We will draw each frame onto this figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.solution = [(-1, -1)]
        self.ax.axis('off')
        self.ax.axis('equal')
        self.done = False
    
    def draw_walls(self):
        """Draws wall around the grid to stop exploring redundant nodes"""
        self.obstacles |= {(i, -2) for i in range(-2, self.width+4)}
        self.obstacles |= {(i, self.height+4) for i in range(-2, self.width+4)}
        self.obstacles |= {(-2, j) for j in range(-2, self.height+5)}
        self.obstacles |= {(self.width+4, j) for j in range(-2, self.height+5)}
        
    def step(self, frame):
        """
        One step of search algorithm.
        Explore a node in the frontier and plot
        all the scatter plots again to create a frame.
        A collection of these frames will be used to
        create the animation by matplotlib.
        """
        # If we are done, don't do anything.
        if self.done:
            return self.sc1, self.sc2, self.sc3, self.sc4, self.sc5, self.sc6
        
        # Run the search algorithm for a single
        # node in the frontier.
        node = self.frontier.pop()
        self.solution = path_states(node)
        if self.is_goal(node.state):
            self.done = True
        else:
            for child in expand(self, node):
                s = child.state
                if s not in self.reached or child.path_cost < self.reached[s].path_cost:
                    self.reached[s] = child
                    self.frontier.add(child)

        # Plot all the new states onto our figure
        # and return them to matplotlib for creating animation.
        self.ax.clear()
        self.ax.axis('off')
        self.ax.axis('equal')
        self.sc1 = self.ax.scatter(*transpose(self.obstacles), marker='s', color='darkgrey')
        self.sc2 = self.ax.scatter(*transpose(list(self.reached)), 1**2, marker='.', c='blue')
        self.sc3 = self.ax.scatter(*transpose(self.solution), marker='s', c='blue')
        self.sc4 = self.ax.scatter(*transpose([node.state]), 9**2, marker='8', c='yellow')
        self.sc5 = self.ax.scatter(*transpose([self.initial]), 9**2, marker='D', c='green')
        self.sc6 = self.ax.scatter(*transpose([self.goal]), 9**2, marker='8', c='red')
        plt.title(f"Explored: {len(self.reached)}, Path Cost: {node.path_cost}\nSolver: {self.solver}")
        return self.sc1, self.sc2, self.sc3, self.sc4, self.sc5, self.sc6
    
    def step_cost(self, s, action, s1):
        return self.cell_weights[s1[0], s1[1]]
        
    def run(self, frames=650):
        """
        Run the main loop of the problem to
        create an animation. If you are running
        on your local machine, you can save animations
        in you system by using the following commands:
        First, you need to download the ffmpeg using:
        Linux/MacOS: `sudo apt install ffmpeg`
        Then you can use the following line of code to generate
        a video of the animation.
        Linux/MacOS : `anim.save('animation.mp4')`
        For Windows users, the process is a little longer:
        Download ffmpeg by following this article: https://www.wikihow.com/Install-FFmpeg-on-Windows
        Then the animation can be saved in a video format as follows:
        Windows: `anim.save('animation.mp4')`
        
        If the animation is not complete, increase the number
        of frames in the below lines of code.
        """
        anim = FuncAnimation(self.fig, self.step, blit=False, interval=100, frames=frames)
        # If you want to save your animations, you can comment either
        # of the lines below.
        # NOTE: FFmpeg is needed to render a .mp4 video of the animation.
        anim.save('/mnt/c/users/tirth/desktop/search_animations/bfs_article_small.mp4')
        # anim.save('animation.html')
        # plt.show()


if __name__ == "__main__":
    random.seed("aima-python")
    # cell_weights = np.random.randint(low=1, high=20, size=3500).reshape(70, 50)
    grid = AnimateProblem(solver='dfs', weight=1.4, height=5, width=10, initial=(1, 4), goal=(9, 4),
                            obstacles=random_lines(X=range(10), Y=range(5), N=3, lengths=range(1, 3)))
    grid.draw_walls()
    grid.run()
