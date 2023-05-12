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
            self.print_board()
        
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

    def print_board(self):
        value_char = {-1:'B', 0:' ', 1:'W', 2:"K"}
        print()
        print(f"Turn of player {self.state.player}")
        print("    0 1 2 3 4 5 6 7 8\n")
        for k in range(len(self.state[1])):
            line = chr(ord("a")+k) + "   "
            line += " ".join([value_char[i] for i in  self.state[1][k]])
            print(line)
