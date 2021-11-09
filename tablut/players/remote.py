from player import Player
from game import calculate_next_state
import json
import asyncio


class __Enemy():
    def __init__(self, address):
        super(__Enemy, self).__init__(self)
        self.data = address

    async def send(self, string):
        pass


class Remote(Player):
    def __init__(self, make_move, enemy_address, board, player):
        super(Remote, self).__init__(self, make_move)
        self.player = player
        self.board = board
        self.enemy = __Enemy(enemy_address)

    def encode(self, action):
        pass

    def decode(self, result):
        pass

    async def next_action_async(self, state, last_action):
        action = self.decode(await self.enemy.send(self.encode(last_action)))
        self.board.select_state(calculate_next_state(self.board.state, action))
        self.make_move(action)

    def next_action(self, state, last_action):
        self.next_action_async(state, last_action)
