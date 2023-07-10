import copy

#from games.game import Game as Gm
infinity = int(1e9)
# example action: (start_row_id, start_column_id, dest_row_id, dest_column_id)
# example state: (player_turn, map)
Board = list[list[int]]
State = tuple[int, Board, int]
Action = tuple[int, int, int, int]


def copy_matrix(matrix: Board):
    return copy.deepcopy(matrix)


class Game():
    ''' Class that contains rules for the Tablut game'''

    black = -1
    white = 1
    king = 2
    __empty_board = [[0,  0,  0, black, black, black,  0,  0,  0],
                     [0,  0,  0,  0, black,  0,  0,  0,  0],
                     [0,  0,  0,  0,  white,  0,  0,  0,  0],
                     [black,  0,  0,  0,  white,  0,  0,  0, black],
                     [black, black,  white,  white,  king,
                      white,  white, black, black],
                     [black,  0,  0,  0,  white,  0,  0,  0, black],
                     [0,  0,  0,  0,  white,  0,  0,  0,  0],
                     [0,  0,  0,  0, black,  0,  0,  0,  0],
                     [0,  0,  0, black, black, black,  0,  0,  0]]

    camp_list = [(0, 3), (0, 4), (0, 5), (1, 4), (8, 3), (8, 4), (8, 5),
                 (7, 4), (3, 0), (4, 0), (5, 0), (4, 1), (3, 8), (4, 8), (5, 8), (4, 7)]
    escape_list = [(0, 1), (0, 2), (0, 6), (0, 7), (8, 1), (8, 2), (8, 6),
                   (8, 7), (1, 0), (2, 0), (6, 0), (7, 0), (1, 8), (2, 8), (6, 8), (7, 8)]
    __player_pieces_values = {"black": [black], "white": [white, king]}
    weight_king = 5

    def create_root(self, player: int, max_game_length=-1_000_000) -> Board:
        return (player, copy_matrix(self.__empty_board), max_game_length)

    def where(self, matrix: Board, condition: list[int]) -> list[tuple[int, int]]:
        return [(i, j) for i in range(len(matrix)) for j in range(len(matrix[i])) if matrix[i][j] in condition]

    def get_piece_positions(self, state: State) -> list[tuple[int, int]]:
        ''' Returns an array containing all the piece position for the current player

        Keyword arguments:
        state -- the state of the game'''
        if state[0] == 1:
            return self.where(state[1], [self.black])
        else:
            return self.where(state[1], [self.white, self.king])

    def get_piece_actions(self, state: State, position: tuple[int, int]) -> list[Action]:
        ''' Returns all the possible action for a specific piece.

        Keyword arguments:
        state -- the state of the board
        position -- tuple that contains row and column for a single piece
        '''
        actions = []

        # general piece
        if state[1][position[0]][position[1]] != self.king:
            # compute movement for row
            if position[1] >= 0 and position[1] <= 8:
                # from player pos-1 to 0 (inclusive)
                for i in reversed(range(0, position[0])):
                    # the action is ok if cell is empty, do not allow camp jump(eg. from left to right camp), go in a camp iff piece is in a camp, don't go in escape cell
                    if state[1][i][position[1]] != 0 or ((i+1, position[1]) not in self.camp_list and (i, position[1]) in self.camp_list) or ((i, position[1]) in self.camp_list and position not in self.camp_list) or (i, position[1]) == (4, 4):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], i, position[1]))
                # from player pos+1 to 8 (inclusive)
                for i in range(position[0]+1, 9):

                    if state[1][i][position[1]] != 0 or ((i-1, position[1]) not in self.camp_list and (i, position[1]) in self.camp_list) or ((i, position[1]) in self.camp_list and position not in self.camp_list) or (i, position[1]) == (4, 4):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], i, position[1]))
            # compute movement for column
            if position[0] >= 0 and position[0] <= 8:
                for i in reversed(range(0, position[1])):
                    if state[1][position[0]][i] != 0 or ((position[0], i+1) not in self.camp_list and (position[0], i) in self.camp_list) or ((position[0], i) in self.camp_list and position not in self.camp_list) or (position[0], i) == (4, 4):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], position[0], i))
                for i in range(position[1]+1, 9):
                    if state[1][position[0]][i] != 0 or ((position[0], i-1) not in self.camp_list and (position[0], i) in self.camp_list) or ((position[0], i) in self.camp_list and position not in self.camp_list) or (position[0], i) == (4, 4):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], position[0], i))
        else:  # if king
            for cardinal in [(position[0]-1, position[1]), (position[0]+1, position[1]), (position[0], position[1]-1), (position[0], position[1]+1)]:
                # direction is possible if don't exit the grid and don't go in castle
                if min(cardinal) >= 0 and max(cardinal) <= 8 and state[1][cardinal[0]][cardinal[1]] == 0 and cardinal != (4, 4) and cardinal not in self.camp_list:
                    actions.append((position[0], position[1], cardinal[0], cardinal[1]))
        return actions

    def actions(self, state: State) -> list[Action]:
        if self.is_terminal(state):
            return []
        actions = []
        for d in self.get_piece_positions(state):
            actions.extend(self.get_piece_actions(state, d))
        return actions

    def result(self, state: State, action: Action) -> State:
        # TODO calculate capture and new state

        # may be wise to disable this check
        actions = self.actions(state)
        if action not in actions:
            raise Exception("Action not allowed", state, action, actions)

        board = copy_matrix(state[1])

        # swap cell
        board[action[0]][action[1]], board[action[2]][action[3]
                                                      ] = board[action[2]][action[3]], board[action[0]][action[1]]
        # compute the value of the two cells in each cardinal direction
        directions = []
        if action[2]-2 >= 0:
            directions.append(
                [(action[2]-1, action[3]), (action[2]-2, action[3])])
        if action[2]+2 <= 8:
            directions.append(
                [(action[2]+1, action[3]), (action[2]+2, action[3])])
        if action[3]-2 >= 0:
            directions.append(
                [(action[2], action[3]-1), (action[2], action[3]-2)])
        if action[3]+2 <= 8:
            directions.append(
                [(action[2], action[3]+1), (action[2], action[3]+2)])

        for cardinal in directions:
            # if king and in castle and sorrounded by 4 sided
            if board[cardinal[0][0]][cardinal[0][1]] == self.king and cardinal[0] == (4, 4) and board[action[2]][action[3]] == self.black and board[cardinal[0][0]-1][cardinal[0][1]] == self.black and board[cardinal[0][0]+1][cardinal[0][1]] == self.black and board[cardinal[0][0]][cardinal[0][1]-1] == self.black and board[cardinal[0][0]][cardinal[0][1]+1] == self.black:
                board[cardinal[0][0]][cardinal[0][1]] = 0
            # if king and in adjacent to the castle and sorrounded by 3 sided
            elif board[cardinal[0][0]][cardinal[0][1]] == self.king and board[action[2]][action[3]] == self.black and cardinal[0] in [(3, 4), (4, 3), (5, 4), (4, 5)] and (board[cardinal[0][0]-1][cardinal[0][1]] == self.black or (cardinal[0][0]-1, cardinal[0][1]) == (4, 4)) and (board[cardinal[0][0]+1][cardinal[0][1]] == self.black or (cardinal[0][0]+1, cardinal[0][1]) == (4, 4)) and (board[cardinal[0][0]][cardinal[0][1]-1] == self.black or (cardinal[0][0], cardinal[0][1]-1) == (4, 4)) and (board[cardinal[0][0]][cardinal[0][1]+1] == self.black or (cardinal[0][0], cardinal[0][1]+1) == (4, 4)):
                board[cardinal[0][0]][cardinal[0][1]] = 0
            # if enemy and near barrier
            elif cardinal[0] not in self.camp_list and not (board[cardinal[0][0]][cardinal[0][1]] == self.king and board[action[2]][action[3]] == self.black and cardinal[0] in [(3, 4), (4, 3), (5, 4), (4, 5), (4, 4)]) and (board[cardinal[0][0]][cardinal[0][1]] != board[action[2]][action[3]] and not (board[action[2]][action[3]] == self.king and board[cardinal[0][0]][cardinal[0][1]] == self.white or board[action[2]][action[3]] == self.white and board[cardinal[0][0]][cardinal[0][1]] == self.king)) and ((board[action[2]][action[3]] == board[cardinal[1][0]][cardinal[1][1]] or (board[action[2]][action[3]] == self.white and board[cardinal[1][0]][cardinal[1][1]] == self.king or board[action[2]][action[3]] == self.king and board[cardinal[1][0]][cardinal[1][1]] == self.white)) or (cardinal[1] == (4, 4) or cardinal[1] in self.camp_list)):
                board[cardinal[0][0]][cardinal[0][1]] = 0

        return ((state[0]+1) % 2, board, state[2] + 1)

    def is_terminal(self, state: State) -> bool:
        if state[2] >= 0:
            return True
        king_pos = self.where(state[1], [self.king])
        if len(king_pos) == 0:
            return True
        king_pos = tuple(king_pos[0])
        return True if min(*king_pos) == 0 or max(*king_pos) == 8 else False

    

    def get_player_pieces_values(self, player: str) -> list[int]:
        ''' Get the type(numerical) of the pieces for a player

        Keyword arguments:
        player -- name of the player
        '''
        if player not in self.__player_pieces_values:
            raise Exception("Player not found")
        return self.__player_pieces_values[player]
    
    def utility(self, state: State, player: str) -> float:
        ''' Return the value of this final state to player

        Keyword arguments:
        state -- the state of the game. The current player is ignored
        player -- the player that want to know the value of a final state
        '''
        assert self.is_terminal(state), "This state is not terminal"
        if state[2] >= 0:
            return -0.1
        
        king_pos = self.where(state[1], [self.king])
        if len(king_pos) == 0:
            return -1 if player == 0 else 1
        return 1 if player == 0 else -1


if __name__ == '__main__':
    # (1, [[0, 0, 0, -1, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, -1, 0], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [-1, -1, 1, 0, 2, 1, 1, 0, -1], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, -1, -1, -1, 0, 0, 0]], -1998), (3, 7, 2, 7), [(3, 7, 3, 6), (3, 7, 3, 5)])
    # state = (1, [[0, 0, 0, -1, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, -1, 0], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [-1, -1, 1, 0, 2, 1, 1, 0, -1], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, -1, -1, -1, 0, 0, 0]], -1998)
    # action = (3,7,2,7)

    # print(Game().actions(state))
    pass
