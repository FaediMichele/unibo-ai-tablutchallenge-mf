import logging
import warnings
import os
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from games.tablut_simple.players.reinforce import Reinforce as R, Model as ModelR
from games.tablut_simple.players.alpha_zero import AlphaTablutZero, Model as ModelZ
from games.tablut_simple.players.alpha_reinforce import AlphaTablutReinforce, Model as ModelZR
from games.tablut_simple.players.console import Console
from games.tablut_simple.game import Game
from games.tablut_simple.console_board import ConsoleBoard
from games.board import Board
from games.player import Player, Action
from functools import partial
import gc
import random
import datetime


state_history = []

def player_manager(board: Board, game: Game, player1: Player, player2: Player,
                   action: Action):
    print(f"Number of possible actions: {len(game.actions(board.state))}")
    current_player = board.state[0]
    if current_player == player2.player:
        player_to_move, opponent = player1, player2
    else:
        player_to_move, opponent = player2, player1

    new_state = game.result(board.state, action)
    board.select_state(new_state)
    state_history.append(new_state)
    if new_state[2] >= 0:
        #player_to_move.end(action, 'draw')
        #opponent.end(action, 'draw')
        print("Game too long")
        # ModelUtil.train_model(player_to_move.model, player_to_move.cache + 
        #                       opponent.cache)
        # ModelUtil.save_model(player_to_move.model)
        
    elif game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
        winner = 'W' if new_state[0] == 1 else 'B'
        print(f"Game ended with a winner: {winner}. "
              f"With {new_state[2]} moves remaining")
        #player_to_move.end(action, opponent.player)
        #opponent.end(action, opponent.player)
    else:
        player_to_move.next_action(action, state_history)

def loaded(board: Board, game: Game, player1: Player, player2: Player):
    ''' When the board have loaded his state, start the first player.
    If the board is a GUI, this function is called on window loaded'''
    if type(board) == ConsoleBoard:
        board.print_board()
    
    state_history.append(board.state)
    if board.state[0] == player1.player:
        player1.next_action(None, state_history)
    else:
        player2.next_action(None, state_history)


def main():
    maximum_turn = 2000
    game = Game()
    board = ConsoleBoard(initial_state=game.create_root(random.randint(0, 1),
                                                        -maximum_turn))
    def make_move(action: Action):
        board.run_manager_function(lambda:
                                   player_manager(board, game,
                                                  player1, player2,
                                                  action))

    model2 = ModelR()
    player1 = Console(make_move, board, game, 1)
    player2 = R(make_move, board, game, 0, model=model2, training=False)

    on_ready = partial(loaded, board, game, player1, player2)

    start_timer = datetime.datetime.now()
    def on_end_of_game():
        # 6 Hours
        if (datetime.datetime.now() - start_timer).total_seconds() < 8800:
            global state_history
            state_history = []
            player1.cache = []
            player2.cache = []
            board.restart(game.create_root(random.randint(0, 1)))
            gc.collect()
            on_ready()

    board.event.on("loaded", on_ready)
    board.event.on("end_of_game", on_end_of_game)
    board.run()

if __name__ == '__main__':
    main()
