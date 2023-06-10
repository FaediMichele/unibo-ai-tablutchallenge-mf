from pymitter import EventEmitter
import os
import asyncio
from functools import partial


package_directory = os.path.dirname(os.path.abspath(__file__))


class Board:
    ''' Class that represent the environment the player are playing.'''

    def __init__(self, initial_state):
        ''' Create the board with an initial state

        Keyword arguments:
        initial_state -- the initial piece position'''
        super().__init__()
        self.event = EventEmitter()
        self.initial_state = initial_state
        self.state = initial_state
        self.scheduler = []

    def select_state(self, state):
        self.state = state
        print(f"Remaining turn to the end: {-state[2]}            ", end="\r")
        
        
    def restart(self, new_board=None):
        if new_board is None:
            self.state = self.initial_state
        else:
            self.state = new_board
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

    
