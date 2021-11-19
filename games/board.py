import numpy as np
from pymitter import EventEmitter
import os

package_directory = os.path.dirname(os.path.abspath(__file__))


class Board:
    ''' Class that represent the environment the player are playing.'''

    def __init__(self, initial_state=np.zeros((9, 9), dtype=np.int32)):
        ''' Create the board with an initial state 

        Keyword arguments:
        initial_state -- the initial piece position'''
        super().__init__()
        self.event = EventEmitter()
        self.state = initial_state
        self.scheduler = []

    def select_state(self, state):
        self.state = state

    def run(self):
        self.event.emit("loaded")
        while len(self.scheduler) > 0:
            print("Scheduler called")
            (self.scheduler.pop())()

    def add_manager_function(self, function):
        self.manager_function = function

    def run_manager_function(self):
        self.scheduler.append(self.manager_function)
