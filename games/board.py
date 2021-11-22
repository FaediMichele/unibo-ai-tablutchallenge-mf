from pymitter import EventEmitter
import os
import asyncio

package_directory = os.path.dirname(os.path.abspath(__file__))


def zeros_matrix(shape):
    return [[0 for _ in range(shape[1])] for _ in range(shape[0])]


class Board:
    ''' Class that represent the environment the player are playing.'''

    def __init__(self, initial_state=zeros_matrix((9, 9))):
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

        async def sched():
            while len(self.scheduler) > 0:
                print("Scheduler called")
                co = (self.scheduler.pop())()
                if asyncio.iscoroutinefunction(co):
                    await co
        asyncio.run(sched())

    def add_manager_function(self, function):
        self.manager_function = function

    def run_manager_function(self):
        self.scheduler.append(self.manager_function)
