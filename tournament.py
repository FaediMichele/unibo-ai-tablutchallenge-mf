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
from games.tablut_simple.players.minmax import MinMax as SMinMax
from games.tablut_simple.game import Game as SimpleGame
from games.tablut_simple.console_board import ConsoleBoard as SimpleConsoleBoard

from games.tablut.players.reinforce import Reinforce as NReinforce, Model as NRModel
from games.tablut.players.alpha_zero import AlphaTablutZero as NAlphaTablutZero, Model as NZModel
from games.tablut.players.minmax import MinMax as NMinMax
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
import json

state_history = []
create_player1 = None
create_player2 = None
player1 = None
player2 = None
results = []
board = None
game = None
result_file = ""

def get_user_inputs():
    parser = argparse.ArgumentParser(description="Game Settings")
    parser.add_argument("--player1_class", choices=["r", "z", "m"], help="Player 1 class (Reinforce or AlphaZero)")
    parser.add_argument("--player1_model_path", help="Player 1 model path", default=None)
    parser.add_argument("--player2_class", choices=["r", "z", "m"], help="Player 1 class (Reinforce or AlphaZero)")
    parser.add_argument("--player2_model_path", help="Player 1 model path", default=None)
    parser.add_argument("--game_type", choices=["n", "s"], help="Type of game")
    parser.add_argument("--show_board", action="store_true", help="Show the board during gameplay")
    parser.add_argument("--max_steps", type=int, help="Maximum number of steps for the game")
    parser.add_argument("--play_time", type=int, help="Number of seconds for the tournament", default=10800)
    parser.add_argument("--thinking_time", type=int, help="Time in milliseconds for a player to think a move")
    parser.add_argument("--result_file", help="Output file to print the results of the matches")

    args = parser.parse_args()
    print(args)
    return (args.player1_class, args.player1_model_path,
            args.player2_class, args.player2_model_path,
            args.game_type, args.max_steps, args.play_time,
            args.thinking_time, args.show_board, args.result_file)

def main():
    global player1, player2, create_player1, create_player2, board, game, result_file

    (player1_class, player1_model_path,
     player2_class, player2_model_path,
     game_type, max_steps, play_time, thinking_time,
     show_board, result_file) = get_user_inputs()
    
    game, board = init_game(game_type, show_board, max_steps)

    create_player1 = partial(init_player, player1_class, player1_model_path, game, board, 0, thinking_time)
    create_player2 = partial(init_player, player2_class, player2_model_path, game, board, 1, thinking_time)
    player1 = create_player1()
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
            board = NormalConsoleBoard(initial_state=game.create_root(0, -max_steps))
        else:
            board = Board(initial_state=game.create_root(0, -max_steps))
    elif game_type == "s":
        game = SimpleGame()
        if show_board:
            board = SimpleConsoleBoard(initial_state=game.create_root(0, -max_steps))
        else:
            board = Board(initial_state=game.create_root(0, -max_steps))
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
        board.restart()
        gc.collect()
        start_game(first_time=False)

    save_results()

def init_player(class_name, model_path, game, board, player_name, ms_for_search):
    if type(game) == NormalGame:
        if class_name == "r":
            model = NRModel(model_path)
            player = NReinforce(None, board, game, player_name, model, True)
        elif class_name == "z":
            model = NZModel(model_path)
            player = NAlphaTablutZero(None, board, game, player_name, model, True, ms_for_search)
        elif class_name == "m":
            player = NMinMax(None, board, game, player_name, ms_for_search)
    
    elif type(game) == SimpleGame:
        if class_name == "r":
            model = SRModel(model_path)
            player = SReinforce(None, board, game, player_name, model, False)
        elif class_name == "z":
            model = SZModel(model_path)
            player = SAlphaTablutZero(None, board, game, player_name, model, False, ms_for_search)
        elif class_name == "m":
            player = SMinMax(None, board, game, player_name, ms_for_search)
    else:
        raise Exception("Game type unknown")
    print(class_name)
    return player


def player_manager(action: Action):
    global player1, player2, board, game, state_history, results

    if board.state[0] == player2.player:
        player_to_move, opponent = player1, player2
    else:
        player_to_move, opponent = player2, player1

    new_state = game.result(board.state, action)
    board.select_state(new_state)
    state_history.append(new_state)

    if new_state[2] >= 0:
        results.append("D")
        print("Game too long - draw")
        
    elif game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
        winner = 'W' if new_state[0] == 1 else 'B'
        results.append(f"{winner}-{-new_state[2]}")
    else:
        player_to_move.next_action(action, state_history)

def save_results():
    global result_file, results, board
    episode_length_list = []
    winner_list = []

    for outcome in results:
        if outcome == "D":
            episode_length_list.append(board.initial_state[2])
            winner_list.append("D")
        else:
            winner, episode_length = outcome.split("-")
            episode_length = int(episode_length)
            winner_list.append(winner)
            episode_length_list.append(episode_length)

    with open(result_file, 'w') as f:
        json.dump({'winners': winner_list, 'episode_length': episode_length_list}, f)

if __name__ == '__main__':
    main()
