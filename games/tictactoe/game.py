import copy

#from games.game import Game as Gm
infinity = int(1e9)
# example action: (start_row_id, start_column_id, dest_row_id, dest_column_id)
# example state: (player_turn, map)
Board = list[list[int]]
State = tuple[int, Board, int]
Action = tuple[int, int]

def copy_matrix(matrix: Board):
    return copy.deepcopy(matrix)



class Game():
    ''' Class that contains rules for the tic tac toe large - 4 aligned'''

    black = -1
    white = 1
    board_size = 9
    __empty_board = [ [0 for _ in range(9)] for _ in range(9)]

    def create_root(self, player: int, max_game_length=-1_000_000) -> Board:
        return (player, copy_matrix(self.__empty_board), max_game_length)

    def get_piece_positions(self, state: State) -> list[tuple[int, int]]:
        ''' Returns an array containing all the piece position for the current player

        Keyword arguments:
        state -- the state of the game'''
        if state[0] == 1:
            return self.where(state[1], [self.black])
        else:
            return self.where(state[1], [self.white])

    def actions(self, state: State) -> list[Action]:
        if self.is_terminal(state):
            return []
        actions = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if state[1][x][y] == 0:
                    actions.append((x, y))
        return actions

    def result(self, state: State, action: Action) -> State:
        player, board, remainin_steps = state

        if action not in self.actions(state):
            raise Exception("Action not valid")
        
        board[action[0]][action[1]] = self.black if player == 1 else self.white

        return ((player + 1) % 2, board, remainin_steps + 1)
    
    def get_connected(self, board: Board) -> int:
        # check row and column
        for i in range(self.board_size):
            count_1 = 0
            last_check_i = -2
            count_k = 0
            last_check_k = -2
            for k in range(self.board_size):
                if board[i][k] != 0:
                    if board[i][k] == last_check_i:
                        count_1 += 1
                        if count_1 >= 3:
                            return last_check_i
                    else:
                        count_1 = 0
                        last_check_i = board[i][k]
                else:
                    count_1 = 0
                    last_check_i = -2
                
                if board[k][i] != 0:
                    if board[k][i] == last_check_k:
                        count_k += 1
                        if count_k >= 3:
                            return last_check_k
                    else:
                        count_k = 0
                        last_check_k = board[k][i]
                else:
                    count_k = 0
                    last_check_k = -2

        # check for oblique lines
        for x in range(self.board_size ):
            for y in range(self.board_size):
                if board[x][y] == 0:
                    continue
                for slide in [[(-k, k) for k in range(0,4)],
                              [(k, -k) for k in range(0,4)],
                              [(k, k) for k in range(0,4)],
                              [(-k, -k) for k in range(0,4)]]:                    
                    for dx, dy in slide:
                        if min(x + dx, y + dy) < 0 or max(x + dx, y + dy) > 8:
                            break
                        if board[x + dx][y + dy] != board[x][y]:
                            break
                    else:
                        return board[x][y]
        return 0

    def is_terminal(self, state: State) -> bool:
        if state[2] >= 0:
            return True
        return self.get_connected(state[1]) != 0

    

    def get_player_pieces_values(self, player: str) -> list[int]:
        ''' Get the type(numerical) of the pieces for a player

        Keyword arguments:
        player -- name of the player
        '''
        return self.white if player == 0 else self.black
    
    def utility(self, state: State, player: str) -> float:
        ''' Return the value of this final state to player

        Keyword arguments:
        state -- the state of the game. The current player is ignored
        player -- the player that want to know the value of a final state
        '''
        assert self.is_terminal(state), "This state is not terminal"

        if state[2] >= 0:
            return 0.1
        
        winning_piece = self.get_connected(state[1])
        if winning_piece == self.white and player == 0:
            return 1
        if winning_piece == self.black and player == 1:
            return 1
        return -1
