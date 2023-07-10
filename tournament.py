import logging
import warnings
import os
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib

from games.board import Board
from games.player import Action, Player
from functools import partial
import gc
import random
import datetime
import argparse
import time
import json

def get_user_inputs():
    parser = argparse.ArgumentParser(description="Game Settings")
    parser.add_argument("--player1_class", "-p1", choices=["r", "z", "rnd", "mm"], help="Player 1 class (Reinforce, AlphaZero, random, MinMax)")
    parser.add_argument("--player1_model_path", help="Player 1 model path", default=None)
    parser.add_argument("--player2_class", "-p2", choices=["r", "z", "rnd", "mm"], help="Player 2 class (Reinforce, AlphaZero, random, MinMax)")
    parser.add_argument("--player2_model_path", help="Player 2 model path", default=None)

    parser.add_argument("--game", '-g', choices=["tablut", "ta", "tictactoe", 'ti'], help="Game to play")
    parser.add_argument("--player_model_path", help="Player 1 model path", default=None)
    parser.add_argument("--remote", '-r', action="store_true", help="Play using a remote server")
    parser.add_argument("--show_board", '-s', action="store_true", help="Show the board during gameplay")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of steps for the game")
    parser.add_argument("--play_time", type=int, help="Number of match for the tournament", default=50)
    parser.add_argument("--thinking_time", type=int, help="Time in milliseconds for a player to think a move")
    parser.add_argument("--node_for_search", type=int, default=None, help="Number of expanded node in the MCTS")
    parser.add_argument("--max_depth_id", type=int, default=None, help="Depth to reach for iterative deepening")
    parser.add_argument("--output_file", type=str, help="Output file for stats")

    args = parser.parse_args()
    return (args.game,
            args.player1_class,
            args.player1_model_path,
            args.player2_class,
            args.player2_model_path,
            args.remote,
            args.max_steps,
            args.play_time,
            args.thinking_time,
            args.show_board,
            args.node_for_search,
            args.max_depth_id,
            args.output_file)

class Tournament:
    state_history = []
    create_player1 = None
    create_player2 = None
    player1 = None
    player2 = None
    results = None
    board = None
    game = None

    def __init__(self,
                 game: str,
                 player1_class: str,
                 player1_model_path: str,
                 player2_class: str,
                 player2_model_path: str,
                 remote: bool,
                 max_steps: int,
                 play_time: int,
                 thinking_time: int,
                 show_board: bool,
                 node_for_search: int,
                 max_depth_id: int,
                 output_file: str) -> None:
        self.output_file = output_file
        self.stats = {'results': [] }
        if game in ["tablut", "ta"]:
            tablut = importlib.import_module('games.tablut')
            MinMaxModule = importlib.import_module('games.tablut.players.minmax')
            AlphaZeroModule = importlib.import_module('games.tablut.players.alpha_zero')
            ReinforceModule = importlib.import_module('games.tablut.players.reinforce')
            self.Game = getattr(tablut, 'Game')
            self.ConsoleBoard = getattr(tablut, 'ConsoleBoard')
            self.ConsoleBoard = getattr(tablut, 'ConsoleBoard')
            self.MinMaxPlayer = getattr(MinMaxModule, 'MinMax')
            self.AlphaZeroPlayer = getattr(AlphaZeroModule, 'AlphaTablutZero')
            self.AlphaZeroModel = getattr(AlphaZeroModule, 'Model')
            self.ReinforcePlayer = getattr(ReinforceModule, 'Reinforce')
            self.ReinforceModel = getattr(ReinforceModule, 'Model')
            self.players_names = ["W", "B"]

        elif game in ["tictactoe", "ti"]:
            tictactoelib = importlib.import_module('games.tictactoe')
            AlphaZeroModule = importlib.import_module('games.tictactoe.players.alpha_zero')
            ReinforceModule = importlib.import_module('games.tictactoe.players.reinforce')
            self.Game = getattr(tictactoelib, 'Game')
            self.ConsoleBoard = getattr(tictactoelib, 'ConsoleBoard')
            self.AlphaZeroPlayer = getattr(AlphaZeroModule, 'AlphaZero')
            self.AlphaZeroModel = getattr(AlphaZeroModule, 'Model')
            self.ReinforcePlayer = getattr(ReinforceModule, 'Reinforce')
            self.ReinforceModel = getattr(ReinforceModule, 'Model')
            self.players_names = ["O", "X"]
            
        self.game = self.Game()
        if show_board:
            self.board = self.ConsoleBoard(initial_state=self.game.create_root(0, -max_steps))
        else:
            self.board = Board(initial_state=self.game.create_root(0, -max_steps))

        self.create_player1 = partial(self.init_player, player1_class, player1_model_path, self.game, self.board, 0, thinking_time, remote, node_for_search, max_depth_id)
        self.create_player2 = partial(self.init_player, player2_class, player2_model_path, self.game, self.board, 1, thinking_time, remote, node_for_search, max_depth_id)
        self.player1 = self.create_player1()
        self.player2 = self.create_player2()

        self.stats['player_1'] = player1_class
        self.stats['player_2'] = player2_class
        self.stats['player_1_model_path'] = player1_model_path
        self.stats['player_2_model_path'] = player2_model_path

        self.board.event.on("loaded", self.loaded)
        self.board.event.on("end_of_game", partial(self.on_end_of_game,
                                            play_time))
        self.played_matches = 0
    def save_stats(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def start(self):
        self.start_game()
        
    def loaded(self):
        if type(self.board) == self.ConsoleBoard:
            self.board.print_board()

        self.state_history.append(self.board.state)
        if self.board.state[0] == self.player1.player:
            self.player1.next_action(None, self.state_history)
        else:
            self.player2.next_action(None, self.state_history)
        
        
    def start_game(self, first_time=True):

        def make_move(action: Action):
            self.board.run_manager_function(lambda: self.player_manager(action))
        self.player1.make_move = make_move
        self.player2.make_move = make_move
        self.state_history = []
        if first_time:
            self.board.run()
        else:
            self.loaded()

    def on_end_of_game(self, total_seconds):
        self.played_matches += 1
        if self.played_matches < total_seconds:
            self.state_history = []
            self.player1 = self.create_player1()
            self.player2 = self.create_player2()
            self.board.restart(self.game.create_root(random.randint(0, 1), self.board.initial_state[2]))
            gc.collect()
            self.start_game(first_time=False)

    def init_player(self, class_name, model_path, game, board, player, ms_for_search, remote, node_for_search, max_depth_id):
        if class_name == "r":
            if model_path is not None:
                model = self.ReinforceModel(model_path, remote=remote)
            else:
                model = self.ReinforceModel(remote=remote)
            player = self.ReinforcePlayer(None, board, game, player, model, False)
        if class_name == "z":
            if model_path is not None:
                model = self.AlphaZeroModel(model_path, remote=remote)
            else:
                model = self.AlphaZeroModel(remote=remote)
            player = self.AlphaZeroPlayer(None, board, game, player, model, False, ms_for_search, node_for_search)
        if class_name == "mm":
            player = self.MinMaxPlayer(None, board, game, player, ms_for_search, max_depth_id)
        if class_name == "rnd":
            player = Player(None, board, game, player)
        return player

    def player_manager(self, action: Action):

        if self.board.state[0] == self.player2.player:
            player_to_move, opponent = self.player1, self.player2
        else:
            player_to_move, opponent = self.player2, self.player1

        new_state = self.game.result(self.board.state, action)
        self.board.select_state(new_state)
        self.state_history.append(new_state)

        if new_state[2] >= 0:
            print("Game too long - draw")
            self.stats['results'].append('D')
        elif self.game.is_terminal(new_state) or len(self.game.actions(new_state)) == 0:
            winner = 'W' if new_state[0] == 1 else 'B'

            print(f"Game ended with a winner: {winner}-{new_state[2]}. "
                f"With {new_state[2]} moves remaining")
            self.stats['results'].append(f"{winner}{new_state[2]}")
        else:
            player_to_move.next_action(action, self.state_history)

if __name__ == '__main__':
    Tournament(*get_user_inputs()).start()