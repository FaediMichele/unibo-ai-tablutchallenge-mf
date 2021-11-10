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

    def select_state(self, state):
        self.state = state

    def run(self):
        self.event.emit("loaded")
