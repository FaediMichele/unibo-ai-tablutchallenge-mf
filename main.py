import os
os.environ['KIVY_NO_ARGS'] = '1'

import argparse
from typing import Type
import logging

from games.player import Player
from games.tablut.board import Board as KivyBoard
from games.board import Board
from games.tablut.game import Game
from games.tablut.players.console import Console
from games.tablut.players.kivy import Kivy
from games.player import Player as RandomPlayer
from games.tablut.players.minmax import MinMax
from games.tablut.players.remote import Remote
import sys
import numpy as np

player_turn = 0
game = None
players = []
action_ready = False
board = False


def main(players_data: list[tuple[str, Type[Player]]],
         turn: int = 0, boardtype: Type[Board] = KivyBoard):
    """Start a game with the given parameters."""
    global player_turn, game, board, players, action_ready
    player_turn = turn

    game = Game()
    board = boardtype(initial_state=game.create_root(player_turn))

    players = []
    action_ready = False
    # players_history = [] # history is not used

    # Populate players
    default_arguments = make_move, board, game

    for p_name, p_type in players_data:
        players.append(p_type(*default_arguments, p_name))

    logging.info(f'players {[p.player for p in players]}')
    # players_history.append([])
    # players_history.append([])

    board.event.on("cell_pressed", cell_pressed)
    board.event.on("loaded", loaded)

    board.run()


def cell_pressed(data):
    pass
    #print(f"cell pressed: {data}. Is playing the {players[player_turn].player}")


def loaded():
    ''' When the board have loaded his state, start the first player.
    If the board is a GUI, this function is called on window loaded'''

    def player_manager():
        global action_ready
        global player_turn
        print("schedule runned")
        if action_ready:
            new_state = game.result(board.state, action_ready)
            board.select_state(new_state)
            # players_history[player_turn].append(action)
            player_turn = (player_turn + 1) % len(players)
            if game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
                print(f"Player {sys.argv[player_turn+1]} wins")
                return
            action = action_ready
            action_ready = False
            players[player_turn].next_action(action)

    board.add_manager_function(player_manager)
    players[0].next_action(None)


def make_move(action):
    """ Function that manage the turns between two player"""
    global action_ready
    action_ready = action
    board.run_manager_function()


def main_cli():
    """Parse command line arguments and start a game using `main`."""
    parser = argparse.ArgumentParser(description='Start a game of tablut')
    parser.add_argument('-g', '--gui', dest='board', action='store_const',
                        const=KivyBoard, default=Board,
                        help='Play with or without a gui?')

    args = parser.parse_args()
    logging.info(f'CLI arguments {args}')

    main([('white', Console), ('black', Console)], boardtype=args.board)


if __name__ == '__main__':
    # Start a local game
    # main([('white', Kivy), ('black', Kivy)])

    # Start a game from cli arguments
    main_cli()
