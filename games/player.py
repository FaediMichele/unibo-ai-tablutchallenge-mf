import random
from games.board import Board
from games.game import Game, State, Action
import logging
from collections.abc import Callable
from typing import Union



class Player:
    ''' Player that take random actions '''

    def __init__(self, make_move: Callable[[Union[None, State, Action]], None], board: Board, game: Game, player: str):
        """Create a new player tha play randomly

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        """
        self.make_move = make_move
        self.board = board
        self.game = game
        self.player = player
        super(Player, self).__init__()

    def next_action(self, last_action: Action, state_history: list[State]):
        ''' Function called when the opponent take the move and now is the turn of this player

        Keyword arguments:
        last_action -- the last move that the opponent have take
        '''
        actions = self.game.actions(self.board.state)
        if len(actions) == 0:
            print(f"Draw. The {self.player} can not do any moves")
            self.make_move(None)
        else:
            self.make_move(actions[random.randint(0, len(actions)-1)])

    def end(self, last_action: Action, opponent, winner: str):
        """Called when a player wins.

        last_action -- the winning move
        opponent -- The opponent player
        winning -- the winning player
        """
        logging.info(f'Calling win on {self.player}, winning: {winner}')
