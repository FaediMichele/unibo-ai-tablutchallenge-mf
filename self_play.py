import logging
import warnings
import os
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from games.tablut_simple.players.reinforce import Reinforce as SReinforce, Model as SRModel
from games.tablut_simple.players.alpha_zero import AlphaTablutZero as SAlphaTablutZero, Model as SZModel
from games.tablut_simple.players.alpha_reinforce import AlphaTablutReinforce as SAlphaTablutReinforce, Model as SARModel
from games.tablut_simple.game import Game as SimpleGame
from games.tablut_simple.console_board import ConsoleBoard as SimpleConsoleBoard
from games.tablut.players.reinforce import Reinforce as NReinforce, Model as NRModel
from games.tablut.players.alpha_zero import AlphaTablutZero as NAlphaTablutZero, Model as NZModel
from games.tablut.players.alpha_reinforce import AlphaTablutReinforce as NAlphaTablutReinforce, Model as NARModel
from games.tablut.game import Game as NormalGame
from games.tablut.console_board import ConsoleBoard as NormalConsoleBoard

from games.game import Game
from games.board import Board
from games.player import Player, Action
from functools import partial
import gc
import random
import datetime
import argparse

state_history = []
create_player1 = None
create_player2 = None
player1 = None
player2 = None
results = None
board = None
game = None

def get_user_inputs():
    parser = argparse.ArgumentParser(description="Game Settings")
    parser.add_argument("--player_class", choices=["r", "z", "zr"], help="Player 1 class (Reinforce or AlphaZero)")
    parser.add_argument("--player_model_path", help="Player 1 model path", default=None)
    parser.add_argument("--game_type", choices=["n", "s"], help="Type of game")
    parser.add_argument("--remote", action="store_true", help="Play using a remote server")
    parser.add_argument("--show_board", action="store_true", help="Show the board during gameplay")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum number of steps for the game")
    parser.add_argument("--play_time", type=int, help="Number of seconds for the tournament", default=10800)
    parser.add_argument("--thinking_time", type=int, help="Time in milliseconds for a player to think a move")

    args = parser.parse_args()
    return (args.player_class, args.player_model_path,
            args.game_type, args.remote, args.max_steps, args.play_time,
            args.thinking_time, args.show_board)

def main():
    global player1, player2, create_player1, create_player2, board, game

    (player_class, player_model_path,
     game_type, remote, max_steps, play_time,
     thinking_time, show_board) = get_user_inputs()
    
    game, board = init_game(game_type, show_board, max_steps)

    create_player1 = partial(init_player, player_class, player_model_path, game, board, 0, thinking_time, remote)
    create_player2 = partial(init_player, player_class, player_model_path, game, board, 1, thinking_time, remote)
    player1 = create_player1()
    print("CREATED PLAYER 1", player1)
    player2 = create_player2()

    board.event.on("loaded", loaded)
    board.event.on("end_of_game", partial(on_end_of_game,
                                          datetime.datetime.now(),
                                          play_time))

    start_game()
    
    
def init_game(game_type, show_board, max_steps):
    if game_type == "n":
        game = NormalGame()
        if show_board:
            board = NormalConsoleBoard(initial_state=game.create_root(
                random.randint(0, 1), -max_steps))
        else:
            board = Board(initial_state=game.create_root(
                random.randint(0, 1), -max_steps))
    elif game_type == "s":
        game = SimpleGame()
        if show_board:
            board = SimpleConsoleBoard(initial_state=game.create_root(
                random.randint(0, 1), -max_steps))
        else:
            board = Board(initial_state=game.create_root(
                random.randint(0, 1), -max_steps))
    else:
        raise Exception("unknown game type")
    return game, board
    
    
    
def loaded():
    global board, game, player1, player2, state_history
    if type(board) == SimpleConsoleBoard or type(board) == NormalConsoleBoard:
        board.print_board()

    state_history.append(board.state)
    if board.state[0] == player1.player:
        player1.next_action(None, state_history)
    else:
        player2.next_action(None, state_history)
    
    
def start_game(first_time=True):
    global player1, player2, game, board, state_history

    def make_move(action: Action):
        board.run_manager_function(lambda: player_manager(action))
    player1.make_move = make_move
    player2.make_move = make_move
    state_history = []
    if first_time:
        board.run()
    else:
        loaded()

def on_end_of_game(start_timer, total_seconds):
    global state_history, player1, player2, create_player1, create_player2, board, game
    if (datetime.datetime.now() - start_timer).total_seconds() < total_seconds:
        state_history = []
        player1 = create_player1()
        player2 = create_player2()
        board.restart(game.create_root(random.randint(0, 1), board.initial_state[2]))
        gc.collect()
        start_game(first_time=False)

def init_player(class_name, model_path, game, board, player, ms_for_search, remote):
    if type(game) == NormalGame:
        if class_name == "ar":
            if model_path is not None:
                model = NARModel(model_path, remote=remote)
            else:
                model = NARModel(remote=remote)
            player = NAlphaTablutReinforce(None, board, game, player, model, True, ms_for_search)
        if class_name == "r":
            if model_path is not None:
                model = NRModel(model_path, remote=remote)
            else:
                model = NRModel(remote=remote)
            player = NReinforce(None, board, game, player, model, True)
        if class_name == "z":
            if model_path is not None:
                model = NZModel(model_path, remote=remote)
            else:
                model = NZModel(remote=remote)
            player = NAlphaTablutZero(None, board, game, player, model, True, ms_for_search)
    elif type(game) == SimpleGame:
        if class_name == "zr":
            if model_path is not None:
                model = SARModel(model_path, remote=remote)
            else:
                model = SARModel(remote=remote)
            player = SAlphaTablutReinforce(None, board, game, player, model, True, ms_for_search)
            print("RETURNED PLAYER", player)
        if class_name == "r":
            if model_path is not None:
                model = SRModel(model_path, remote=remote)
            else:
                model = SRModel(remote=remote)
            player = SReinforce(None, board, game, player, model, True)
        if class_name == "z":
            if model_path is not None:
                model = SZModel(model_path, remote=remote)
            else:
                model = SZModel(remote=remote)
            player = SAlphaTablutZero(None, board, game, player, model, True, ms_for_search)
    else:
        raise Exception("Game type unknown")
    return player


def player_manager(action: Action):
    global player1, player2, board, game, state_history

    if board.state[0] == player2.player:
        player_to_move, opponent = player1, player2
    else:
        player_to_move, opponent = player2, player1

    new_state = game.result(board.state, action)
    board.select_state(new_state)
    state_history.append(new_state)

    if new_state[2] >= 0:
        print("Game too long - draw")
        player_to_move.end(action, opponent, 'draw')
        
        player_to_move.model.player_wins('D', new_state[2])
        player_to_move.model.save_model()
        
    elif game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
        winner = 'W' if new_state[0] == 1 else 'B'
        print(f"Game ended with a winner: {winner}. "
            f"With {new_state[2]} moves remaining")
        player_to_move.end(action, opponent, opponent.player)
        
        player_to_move.model.player_wins(winner, new_state[2])
        player_to_move.model.save_model()

    else:
        player_to_move.next_action(action, state_history)

if __name__ == '__main__':
    main()