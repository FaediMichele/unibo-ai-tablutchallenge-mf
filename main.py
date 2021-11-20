from games.tablut.board import Board
from games.tablut.game import Game
from games.tablut.players.console import Console
from games.tablut.players.kivy import Kivy
from games.player import Player as RandomPlayer
from games.tablut.players.minmax import MinMax
import sys
import numpy as np


if __name__ == '__main__':
    # if the argument of the program is not passed start a random player
    player_turn = 0
    if len(sys.argv) == 1:
        sys.argv.append("white")
        sys.argv.append("black")
    elif sys.argv[1].lower() == "black":
        player_turn = 1

    game = Game()
    board = Board(initial_state=game.create_root(player_turn))

    players = []
    action_ready = False
    # players_history = [] # history is not used

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
            if action_ready != False:
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

    players.append(Kivy(make_move, board, game, sys.argv[1]))
    players.append(MinMax(make_move, board, game, sys.argv[2]))
    print(players[0].player, players[1].player)
    # players_history.append([])
    # players_history.append([])

    board.event.on("cell_pressed", cell_pressed)
    board.event.on("loaded", loaded)

    board.run()
