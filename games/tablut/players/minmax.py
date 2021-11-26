from games.player import Player
import math
import random
from games.board import Board as Bd
from games.game import Game
from collections.abc import Callable
infty = math.inf
Board = list[list[int]]
State = tuple[str, Board]
Action = tuple[int, int, int, int]


def cutoff_depth(d: int):
    """A cutoff function that searches to depth d."""
    return lambda game, state, depth: depth > d


def cache(function):
    '''Like lru_cache(None), but only considers the first argument of function.'''
    cache = {}

    def wrapped(x, *args):
        x_hash = (x[0], tuple([tuple(x[1][k]) for k in range(len(x[1]))]))
        if x_hash not in cache:
            cache[x_hash] = function(x, *args)
        return cache[x_hash]
    return wrapped


class MinMax(Player):
    ''' Class for a local player. Is based on a GUI, so if is not present this class may not work.'''

    def __init__(self, make_move, board: Bd, game: Game, player: str, cutoff=cutoff_depth(2), h: Callable[[State, str], float] = None):
        ''' Create a local player

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the player identification. Can be a number or anything else
        max_depth -- the max depth for the min max tree
        '''
        super(MinMax, self).__init__(make_move, board, game, player)
        self.cutoff = cutoff
        self.h = h if h != None else game.h

    def next_action(self, last_action: Action):
        game = self.game
        player = self.player
        print(player)

        @ cache
        def max_value(state: State, alpha: float, beta: float, depth: int):
            if game.is_terminal(state):
                return game.utility(state, player), None

            if self.cutoff(game, state, depth):
                return self.h(state, player), None

            v, move = -infty, None
            actions = game.actions(state)
            for a in actions:
                v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
                if v2 > v or (v2 == v and random.choice((0, 1)) == 1):
                    v, move = v2, a
                    alpha = max(alpha, v)
                if v >= beta:
                    return v, move
            return v, move

        @ cache
        def min_value(state: State, alpha: float, beta: float, depth: int):
            if game.is_terminal(state):
                return game.utility(state, player), None

            if self.cutoff(game, state, depth):
                return self.h(state, player), None

            v, move = +infty, None
            for a in game.actions(state):
                v2, _ = max_value(game.result(state, a), alpha, beta, depth+1)
                if v2 < v:
                    v, move = v2, a
                    beta = min(beta, v)
                if v <= alpha:
                    return v, move
            return v, move
        print(
            f"Player: {self.player} is thinking...")
        _, a = max_value(self.board.state, -infty, infty, 0)
        self.make_move(a)
