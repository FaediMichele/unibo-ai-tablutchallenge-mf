from games.tablut.players.alpha_tablut_zero import AlphaTablutZero
from games.tablut.players.console import Console
from games.tablut.game import Game
from games.board import Board
from games.player import Player, Action
from functools import partial
import logging
import random

state_history = []

def player_manager(board: Board, game: Game, player1: Player, player2: Player,
                   action: Action):
    current_player = board.state[0]
    if current_player == player1.player:
        player_to_move, opponent = player1, player2
    else:
        player_to_move, opponent = player2, player1

    new_state = game.result(board.state, action)
    board.select_state(new_state)
    state_history.append(new_state)
    if game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
        player_to_move.end(action, opponent.player)
        opponent.end(action, opponent.player)
    player_to_move.next_action(action, state_history)

def loaded(board: Board, game: Game, player1: Player, player2: Player):
    ''' When the board have loaded his state, start the first player.
    If the board is a GUI, this function is called on window loaded'''
    if board.state[0] == player1.player:
        player1.next_action(None, [])
    else:
        player2.next_action(None, [])


def main():
    game = Game()
    board = Board(initial_state=game.create_root(random.randint(0, 1)))
    def make_move(action: Action):
        board.run_manager_function(lambda: player_manager(board, game, player1, player2, action))

    player1 = AlphaTablutZero(make_move, board, game, 0, train=True)
    player2 = AlphaTablutZero(make_move, board, game, 1, train=True)

    board.event.on("loaded", partial(loaded, board, game, player1, player2))
    board.run()

if __name__ == '__main__':
    main()
