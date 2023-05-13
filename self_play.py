from games.tablut.players.alphazero.alpha_tablut_zero import AlphaTablutZero, ModelUtil, TransferableMemory
from games.tablut.players.console import Console
from games.tablut.game import Game
from games.tablut.console_board import ConsoleBoard
from games.board import Board
from games.player import Player, Action
from functools import partial
import gc
import random

state_history = []

def player_manager(board: Board, game: Game, player1: Player, player2: Player,
                   action: Action, maximum_turn: int=-1):
    current_player = board.state[0]
    if current_player == player1.player:
        player_to_move, opponent = player1, player2
    else:
        player_to_move, opponent = player2, player1

    assert player_to_move.memory != opponent.memory

    new_state = game.result(board.state, action)
    board.select_state(new_state)
    state_history.append(new_state)
    if len(state_history) > maximum_turn:
        player_to_move.end(action, opponent.player)
        opponent.end(action, player_to_move.player)
        print("Game too long")
        ModelUtil.train_model(player_to_move.model, player_to_move.memory.cache + 
                              opponent.memory.cache)
        ModelUtil.save_model(player_to_move.model)
        
    elif game.is_terminal(new_state) or len(game.actions(new_state)) == 0:
        player_to_move.end(action, opponent.player)
        opponent.end(action, opponent.player)
        print("Game ended with a winner")
        ModelUtil.train_model(player_to_move.model, player_to_move.memory.cache + 
                              opponent.memory.cache,
                              winner=opponent.player)
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
    maximum_turn = 200
    game = Game()
    board = ConsoleBoard(initial_state=game.create_root(random.randint(0, 1)))
    def make_move(action: Action):
        board.run_manager_function(lambda: player_manager(board, game,
                                                          player1, player2,
                                                          action, maximum_turn))

    player1 = AlphaTablutZero(make_move, board, game, 0, memory=TransferableMemory(),
                              greedy=False, ms_for_search=5000,
                              max_moves_for_game=maximum_turn)
    player2 = AlphaTablutZero(make_move, board, game, 1, memory=TransferableMemory(),
                              greedy=False, ms_for_search=5000,
                              max_moves_for_game=maximum_turn)

    on_ready = partial(loaded, board, game, player1, player2)

    def on_end_of_game():
        global state_history
        state_history = []
        player1.cache = []
        player2.cache = []
        player1.memory.reset()
        player2.memory.reset()
        gc.collect()
        board.restart(game.create_root(random.randint(0, 1)))
        on_ready()

    board.event.on("loaded", on_ready)
    board.event.on("end_of_game", on_end_of_game)
    board.run()

if __name__ == '__main__':
    main()
