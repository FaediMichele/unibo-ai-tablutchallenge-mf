import logging
from threading import Thread, Event

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
    '''Like lru_cache(None), but only considers the first and last argument.'''
    cache = {}

    def wrapped(x, *args):
        x_hash = (x[0], tuple([tuple(x[1][k]) for k in range(len(x[1]))]),
                  args[2])
        if x_hash not in cache:
            cache[x_hash] = function(x, *args)
        return cache[x_hash]
    return wrapped


class MinMax(Player):
    '''Automatic player that uses a minimax pruned algorithm.'''

    def __init__(self, make_move, board: Bd, game: Game, player: str, timeout: int, h: Callable[[State, str], float] = None):
        ''' Create a local player

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the player identification. Can be a number or anything else
        timeout -- the max time allowed for each move
        '''
        if timeout <= 5:
            raise ValueError('Timeout should be higher than 5')

        logging.info(f'MinMax timeout {timeout}')

        super(MinMax, self).__init__(make_move, board, game, player)
        self.timeout = timeout
        self.timeout_event = Event()
        self.cached_moves = []
        self.h = h if h != None else game.h

    def next_action(self, last_action: Action, state_history: list[State]):
        """Start a monitored thread for the next action."""
        self.cached_moves.clear()
        self.timeout_event.clear()

        thread = Thread(
            target=self._iterative_deepening,
            args=(last_action,))

        thread.start()

        # Wait for the thread
        thread.join(self.timeout - 5)

        logging.info(f'Timeout triggered')

        # Kill the thread
        self.timeout_event.set()

        # Make last cached move
        self.make_move(self.cached_moves[-1])

    def _iterative_deepening(self, last_action: Action):
        """Perform interative deepening."""
        try:
            depth = 1
            while True:
                self._next_action_cutoff(last_action, cutoff_depth(depth))
                depth += 1
        except TimeoutError:
            return

    def _next_action_cutoff(self, last_action: Action, cutoff: Callable[[Game, State, int], bool]):
        """Calculate and cache the best action with the given cutoff.

        This function is used internally for iterative deepening. The
        result is cached in the list `cached_moves`.
        """
        game = self.game
        player_id = self.board.state[0]
        print(player_id)

        @ cache
        def max_value_root(state: State, alpha: float, beta: float, depth: int):
            if game.is_terminal(state):
                return game.utility(state, player_id), None

            if cutoff(game, state, depth):
                return self.h(state, player_id, False), None

            v, move = -infty, None
            actions = game.actions(state)
            for a in actions:
                v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
                if v2 > v:
                    v, move = v2, a
                    alpha = max(alpha, v)
                if v >= beta:
                    return v, move
            return v, move

        @ cache
        def max_value(state: State, alpha: float, beta: float, depth: int):
            if self.timeout_event.is_set():
                raise TimeoutError

            if game.is_terminal(state):
                return game.utility(state, player_id), None

            if cutoff(game, state, depth):
                return self.h(state, player_id, False), None

            v, move = -infty, None
            actions = game.actions(state)
            for a in actions:
                v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
                if v2 > v:
                    v, move = v2, a
                    alpha = max(alpha, v)
                if v >= beta:
                    return v, move
            return v, move

        @ cache
        def min_value(state: State, alpha: float, beta: float, depth: int):
            if self.timeout_event.is_set():
                raise TimeoutError

            if game.is_terminal(state):
                return game.utility(state, player_id), None

            if cutoff(game, state, depth):
                return self.h(state, player_id, True), None

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
        _, a = max_value_root(self.board.state, -infty, infty, 0)
        self.cached_moves.append(a)
