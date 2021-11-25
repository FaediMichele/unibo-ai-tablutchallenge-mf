import sys
from games.tablut.players.remote import Remote
from games.tablut.players.minmax import MinMax
from games.tablut.game import Game
from games.tablut.players.console import Console
# from games.tablut.board import Board as KivyBoard
# from games.tablut.players.kivy import Kivy
from games.player import Player as RandomPlayer
from games.board import Board
from games.player import Player
import importlib
import random
import logging
from typing import Type
import argparse
import os
os.environ['KIVY_NO_ARGS'] = '1'


# Workaround for kivy automatic window creation:
# Kivy would crash instantly in headless mode due to the window
# creation, hence I protect the imports under these global fields
# and an init function, that will be called only if kivy is wanted
Kivy = None
KivyBoard = None


def init_kivy():
    global Kivy, KivyBoard
    board_module = importlib.import_module('games.tablut.board')
    KivyBoard = board_module.Board

    player_module = importlib.import_module('games.tablut.players.kivy')
    Kivy = player_module.Kivy


player_turn = 0
game = None
players = []
action_ready = False
state_ready = False
board = False


# Used by argparse to parse cli arguments
TURNS_MAP = {'white': 0, 'black': 1}


def main(players_data: list[tuple[str, Type[Player], tuple]],
         turn: int = 0, boardtype: Type[Board] = Board):
    """Start a game with the given parameters.

    `players_data` is the list of players in the following format:
    [(name, PlayerType, (arg1, ...)), ...]
    The additional arguments will be passed to the player's constructor.
    """
    global player_turn, game, board, players, action_ready, state_ready
    player_turn = turn
    print(boardtype)
    game = Game()
    board = boardtype(initial_state=game.create_root(player_turn))

    players = []
    action_ready = False
    state_ready = False
    # players_history = [] # history is not used

    # Populate players
    default_arguments = make_move, board, game

    for p_name, p_type, params in players_data:
        players.append(p_type(*default_arguments, p_name, *params))

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
        global action_ready, player_turn, state_ready
        print("schedule runned")
        if action_ready:
            new_state = game.result(board.state, action_ready)
            board.select_state(new_state)
            # players_history[player_turn].append(action)

            if game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
                print(f"Player {players[player_turn].player} wins")
                return
            player_turn = (player_turn + 1) % len(players)
            action = action_ready
            action_ready = False
            players[player_turn].next_action(action)
        elif state_ready:
            board.select_state(state_ready)
            # players_history[player_turn].append(action)

            if game.is_terminal(state_ready) or len(game.actions(state_ready)) == 0:
                print(f"Player {players[player_turn].player} wins")
                return
            player_turn = (player_turn + 1) % len(players)
            action = action_ready
            action_ready = False
            players[player_turn].next_action(action)

    board.add_manager_function(player_manager)
    players[0].next_action(None)


def make_move(param):
    """ Function that manage the turns between two player"""
    global action_ready, state_ready
    if type(param[1]) == tuple:
        action_ready = param
        state_ready = False
    else:
        state_ready = param
        action_ready = False

    board.run_manager_function()


def main_cli():
    """Parse command line arguments and start a game using `main`."""
    parser = argparse.ArgumentParser(description='Start a game of tablut')
    parser.add_argument('-g', '--gui', dest='gui', action='store_true',
                        help='play with or without a gui?')
    parser.add_argument('-s', '--starts', dest='turn', nargs=1,
                        default=['white'],
                        help='the player who shall start. Supported options '
                              'are: white, black, random. Default behaviour '
                              'is: white.')

    args = parser.parse_args()
    logging.info(f'CLI arguments {args}')

    if args.gui:
        init_kivy()
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Manage first player turn
    parsed_turn, = args.turn
    turn = random.choice((0, 1))
    if parsed_turn != 'random':
        turn = TURNS_MAP[parsed_turn]

    # Manage player types (TODO manage remote, minimax, etc.) and order
    playertype = Kivy if args.gui else Console
    players_ = [('white', playertype, tuple()), ('black', playertype, tuple())]
    if turn:        # If it's black's turn
        players_.reverse()

    main(players_, boardtype=KivyBoard if args.gui else Board)


if __name__ == '__main__':
    # Start a local game
    init_kivy()
    main([('white', Kivy, tuple()), ('black', Remote, (('127.0.0.1', 5800),))],
         boardtype=KivyBoard)

    # Start a game against minimax
    # init_kivy()
    # main([('white', Kivy, tuple()), ('black', MinMax, tuple())],
    #      boardtype=KivyBoard)

    # Start a game from cli arguments
    # main_cli()
