from pymitter import EventEmitter
import os
import asyncio
from functools import partial


package_directory = os.path.dirname(os.path.abspath(__file__))


def zeros_matrix(shape):
    return [[0 for _ in range(shape[1])] for _ in range(shape[0])]


class Board:
    ''' Class that represent the environment the player are playing.'''

    def __init__(self, initial_state=zeros_matrix((9, 9)), show_board=False):
        ''' Create the board with an initial state

        Keyword arguments:
        initial_state -- the initial piece position'''
        super().__init__()
        self.event = EventEmitter()
        self.initial_state = initial_state
        self.state = initial_state
        self.scheduler = []
        self.show_board = show_board

    def select_state(self, state):
        self.state = state
        if self.show_board:
            self.print_board(state)
        
    def restart(self, new_board=None):
        if new_board is None:
            self.state = self.initial_state
        else:
            self.state = self.initial_state
        self.scheduler = []

    def run(self):
        self.event.emit("loaded")

        async def sched():
            while len(self.scheduler) > 0:
                while len(self.scheduler) > 0:
                    co = (self.scheduler.pop())()
                    if asyncio.iscoroutinefunction(co):
                        await co
                self.event.emit('end_of_game')
        asyncio.run(sched())

    def run_manager_function(self, func):
        self.scheduler.append(func)

    def print_board(self, state):
        print()
        print("    0  1  2  3  4  5  6  7  8")
        for k in range(len(state[1])):
            line = chr(ord("a")+k)+"   "
            for i in range(len(state[1][k])):
                line += str(state[1][k][i]) + \
                    ("  " if state[1][k][i] >= 0 else " ")
            print(line)
