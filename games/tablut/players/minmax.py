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

black = -1
white = 1
king = 2
distance_from_excapes = [[0, 0, 0, 1, 2, 1, 0, 0, 0],
                               [0, 1, 1, 2, 3, 2, 1, 1, 0],
                               [0, 1, 2, 3, 4, 3, 2, 1, 0],
                               [1, 2, 3, 4, 5, 4, 3, 2, 1],
                               [2, 3, 4, 5, 6, 5, 4, 3, 2],
                               [1, 2, 3, 4, 5, 4, 3, 2, 1],
                               [0, 1, 2, 3, 4, 3, 2, 1, 0],
                               [0, 1, 1, 2, 3, 2, 1, 1, 0],
                               [0, 0, 0, 1, 2, 1, 0, 0, 0]]

weight_heuristic = {1: {"king": 10, "soldier": 1, "king_position": 30},
                    0: {"king": 5, "soldier": 10, "king_position": 20}}
player_pieces_values = {"black": [black], "white": [white, king]}

def where(matrix: Board, condition: list[int]) -> list[tuple[int, int]]:
    return [(i, j) for i in range(len(matrix)) for j in range(len(matrix[i])) if matrix[i][j] in condition]


def heuristic( state: State, player: int, min_max: bool) -> float:
    num_white = len(where(state[1], [white]))
    num_black = len(where(state[1], [black]))
    king_pos = where(state[1], [king])[0]
    enemy_adjacent_king = sum([1 if state[1][around_king[0]][around_king[1]] == black else 0
                               for around_king in [(king_pos[0] - 1, king_pos[1]),
                                                   (king_pos[0] + 1, king_pos[1]),
                                                   (king_pos[0], king_pos[1] - 1),
                                                   (king_pos[0], king_pos[1] + 1)]])
    player_index = -2 * player + 1
    
    if (player == 0 and not min_max) or (player == 1 and min_max):
        min_max_player = 1
    else:
        min_max_player = 0

    soldier_value = (num_white - num_black)
    king_value = (enemy_adjacent_king if king_pos == (4, 4) else # castle
                  enemy_adjacent_king * 2 if king_pos not in [(3, 4), (4, 3), (5, 4), (4, 5)] else # near the castle
                  enemy_adjacent_king * 4) # else
    king_pos_value = distance_from_excapes[king_pos[0]][king_pos[1]]

    return player_index * (weight_heuristic[min_max_player]["soldier"] * soldier_value
                           - weight_heuristic[min_max_player]["king"] * king_value
                           - weight_heuristic[min_max_player]["king_position"] * king_pos_value)

def final_state_value(state: State, player: int) -> float:
    king_pos = where(state[1], [king])
    if len(king_pos) != player:
        return 100_000
    else:
        return -100_000

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

    def __init__(self, make_move, board: Bd, game: Game, player: str, timeout: int):
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

    def next_action(self, last_action: Action, state_history: list[State]):
        """Start a monitored thread for the next action."""
        self.cached_moves.clear()
        self.timeout_event.clear()

        thread = Thread(
            target=self._iterative_deepening,
            args=(last_action,))

        thread.start()
        # Wait for the thread
        thread.join(self.timeout / 1000)
        logging.info(f'Timeout triggered')

        # Kill the thread
        self.timeout_event.set()

        # Make last cached move
        if random.random() > 0.9:
            ls = list({k: 0 for k in self.cached_moves}.keys())
            if len(ls) <= 1:
                self.make_move(random.choice(self.game.actions(self.board.state)))
            else:
                self.make_move(ls[-2])
            print("suboptimal action")
        else:
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
        adversary_id = player_id + 1 % 2
        @ cache
        def max_value_root(state: State, alpha: float, beta: float, depth: int):
            if game.is_terminal(state):
                raise Exception("Initial state can not be final")

            if cutoff(game, state, depth):
                return heuristic(state, player_id, False), None

            v, move = -infty, None
            for a in game.actions(state):
                v2, _ = min_value(game.result(state, a), alpha, beta, depth + 1)
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
                return final_state_value(state, player_id), None

            if cutoff(game, state, depth):
                return heuristic(state, player_id, False), None

            v, move = -infty, None
            for a in game.actions(state):
                v2, _ = min_value(game.result(state, a), alpha, beta, depth + 1)
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
                return final_state_value(state, player_id), None

            if cutoff(game, state, depth):
                return heuristic(state, adversary_id, True), None

            v, move = +infty, None
            for a in game.actions(state):
                v2, _ = max_value(game.result(state, a), alpha, beta, depth + 1)
                if v2 < v:
                    v, move = v2, a
                    beta = min(beta, v)
                if v <= alpha:
                    return v, move
            return v, move
        _, a = max_value_root(self.board.state, -infty, infty, 0)
        self.cached_moves.append(a)
