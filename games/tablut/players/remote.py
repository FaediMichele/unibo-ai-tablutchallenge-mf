from games.player import Player
from game import calculate_next_state
import json
import asyncio


class __Enemy():
    ''' Class the encapsulate the send and receive of commands for the remote player/server '''

    def __init__(self, address):
        super(__Enemy, self).__init__(self)
        self.data = address

    async def send(self, string):
        pass


class Remote(Player):
    ''' Class for a remote player '''

    def __init__(self, make_move, board, game, player, enemy_address):
        ''' Create a remote player.

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        enemy_address -- the data for comunicatin with the remote player ( #TODO yet to be defined )
        '''
        super(Remote, self).__init__(self, make_move, board, game, player)
        self.enemy = __Enemy(enemy_address)

    def encode(self, action):
        ''' Parse an action to the server comunication format'''
        pass

    def decode(self, result):
        ''' Parse a server comunication format to an action'''
        pass

    async def next_action_async(self, last_action):
        action = self.decode(await self.enemy.send(self.encode(last_action)))
        self.board.select_state(calculate_next_state(self.board.state, action))
        self.make_move(action)

    def next_action(self, last_action):
        self.next_action_async(self.board.state, last_action)
