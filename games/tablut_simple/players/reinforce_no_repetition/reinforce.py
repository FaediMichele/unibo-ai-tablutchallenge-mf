from games.player import Player
from games.board import Board
from games.tablut_simple.console_board import ConsoleBoard
from games.tablut_simple.game import Game, State, Action
from .util import state_hashable, argmax, unhash_state, policy_matrix_to_policy, PREVIOUS_STATE_TO_MODEL
from .model import Model
from typing import Callable
import tensorflow as tf
import datetime
import random
import logging

class Reinforce(Player):
    ''' Player that take random actions '''

    def __init__(self, make_move: Callable[[None | State |Action], None],
                 board: Board,
                 game: Game,
                 player: int,
                 model: Model,
                 training: bool=False):
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
        self.cache = []
        self.tree = None
        self.training = training
        self.model = model
        super(Player, self).__init__()

    def next_action(self, last_action: Action, state_history: list[State]):
        ''' Function called when the opponent take the move and now is the
        turn of this player

        Keyword arguments:
        last_action -- the last move that the opponent have take
        '''
        state = self.board.state
        actions = tuple(self.game.actions(state))
        policy, state_value, state_tensor = self._evaluate_state(state, actions, state_history, True)
        
        if self.training:
            # the if is true when the model assign 0 probability to all
            # possible moves
            if sum(policy) < 0.001:
                action_taken = random.choices(actions)[0]
            else:
                try:
                    action_taken = random.choices(actions, policy)[0]
                except:
                    print(sum(policy))
                    action_taken = random.choices(actions)[0]

            self.cache.append([state_tensor, action_taken])
            self.make_move(action_taken)
            print(f"{self.player} think that the state value is {state_value}")
        else:
            
            self.make_move(actions[argmax(range(len(policy)), key=lambda x: policy[x])])

    def end(self, last_action: Action, opponent: Player, winner: str):
        """Called when a player wins.

        last_action -- the winning move
        winning -- the winning player
        """
        logging.info(f'Calling win on {Player}, winning: {winner}')
        if self.training:
            if winner == 'draw':
                g = 0
            elif winner == self.player: # actually never happen
                g = 1.0
            else:
                g = -1.0
        zs = [g for _ in self.cache] + [-g for _ in opponent.cache]
        self.model.save_cache([(self.cache, g)])
        self.model.save_cache([(opponent.cache, -g)])
        self.model.train_episode(self.cache + opponent.cache, zs)
        

    def _evaluate_state(self, state: State, actions: list[Action],
                        state_history: list[State],
                        return_state_transformed=False) -> tuple[list[float], float]:
        if self.game.is_terminal(state):
            if state[2] >= 0:
                state_value = 0
            elif state[0] == self.player:
                state_value = -1
            else:
                state_value = 1

            if return_state_transformed:
                return [], state_value, self._transform(state, state_history)
            else:
                return [], state_value

        state_transformed = self._transform(state, state_history)
        state_value, policy = self.model.predict(state_transformed)
        policy = policy_matrix_to_policy(policy, actions)

        if return_state_transformed:
            return policy, state_value, state_transformed
        else:
            return policy, state_value

    def _transform(self, state: State, state_history: list[State]) -> tf.Tensor:
        player, board, _ = state
        boards = [sh[1] for sh in state_history[-PREVIOUS_STATE_TO_MODEL - 1:]]
        while len(boards) <= PREVIOUS_STATE_TO_MODEL:
            boards.insert(0, boards[0])
            
        ###
        # for b in boards:
        #     ConsoleBoard().select_state((1, b, -10))
        # input()
        ###
        boards = list(reversed(boards))
        board_shape = (len(board), len(board[0]), 1)
        
        tensor_board = tf.stack(boards, axis=2)
        tensor_state_player = (tf.ones(board_shape, tensor_board.dtype) *
                               (player * 2 - 1))
        return tf.concat([tensor_board, tensor_state_player], axis=2)
