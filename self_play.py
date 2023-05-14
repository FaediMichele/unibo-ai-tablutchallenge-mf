from games.tablut.players.alphazero.alpha_tablut_zero import AlphaTablutZero
from games.tablut.players.alphazero.model import ModelUtil
from games.tablut.game import Game
from games.tablut.console_board import ConsoleBoard
from games.board import Board
from games.player import Player, Action
from functools import partial
import gc
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
    if new_state[2] >= 0:
        player_to_move.end(action, opponent.player)
        opponent.end(action, player_to_move.player)
        print("Game too long")
        # ModelUtil.train_model(player_to_move.model, player_to_move.cache + 
        #                       opponent.cache)
        # ModelUtil.save_model(player_to_move.model)
        
    elif game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
        player_to_move.end(action, opponent.player)
        opponent.end(action, opponent.player)
        print("Game ended with a winner")
        ModelUtil.train_model(player_to_move.model,
                              player_to_move.cache +
                              opponent.cache,
                              winner='W' if new_state[0] == 1 else 'B')
        ModelUtil.save_model(player_to_move.model)
    else:
        player_to_move.next_action(action, state_history)

def loaded(board: Board, game: Game, player1: Player, player2: Player):
    ''' When the board have loaded his state, start the first player.
    If the board is a GUI, this function is called on window loaded'''
    if type(board) == ConsoleBoard:
        board.print_board()
    if board.state[0] == player1.player:
        player1.next_action(None, [])
    else:
        player2.next_action(None, [])


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

    player1 = AlphaTablutZero(make_move, board, game, 0,
                              greedy=False, ms_for_search=10000)
    player2 = AlphaTablutZero(make_move, board, game, 1,
                              greedy=False, ms_for_search=10000)

    on_ready = partial(loaded, board, game, player1, player2)

    def on_end_of_game():
        global state_history
        state_history = []
        player1.cache = []
        player2.cache = []
        gc.collect()
        board.restart(game.create_root(random.randint(0, 1)))
        on_ready()

    board.event.on("loaded", on_ready)
    board.event.on("end_of_game", on_end_of_game)
    board.run()

if __name__ == '__main__':
    main()
