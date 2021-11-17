from games.tablut.board import Board
from games.tablut.game import Game
from games.tablut.players.local import Local
from games.player import Player as RandomPlayer
import sys
import numpy as np


if __name__ == '__main__':
    # if the argument of the program is not passed start a random player
    if len(sys.argv) == 1:
        if np.random.randint(0, 1) == 0:
            sys.argv.append("black")
            sys.argv.append("white")
        else:
            sys.argv.append("white")
            sys.argv.append("black")

    game = Game()
    board = Board(initial_state=game.create_root(sys.argv[1]))
    player_turn = 0
    players = []
    # players_history = [] # history is not used

    def cell_pressed(data):
        pass
        #print(f"cell pressed: {data}. Is playing the {players[player_turn].player}")

    def loaded():
        ''' When the board have loaded his state state the first player.
        If the board is a GUI, this function is called on window loaded'''
        players[0].next_action(None)

    def make_move(action):
        """ Function that manage the turns between two player"""
        global player_turn
        new_state = game.result(board.state, action)
        board.select_state(new_state)
        # players_history[player_turn].append(action)
        player_turn = (player_turn + 1) % len(players)
        players[player_turn].next_action(action)

    players.append(RandomPlayer(make_move, board, game, sys.argv[1]))
    players.append(Local(make_move, board, game, sys.argv[2]))
    # players_history.append([])
    # players_history.append([])

    board.event.on("cell_pressed", cell_pressed)
    board.event.on("loaded", loaded)

    board.run()
