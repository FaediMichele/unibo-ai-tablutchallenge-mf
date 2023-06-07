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
    __empty_board = [[0, 0, black, black, black, 0, 0],
                     [0, 0, 0,  white, 0, 0, 0],
                     [black, 0, 0, white, 0, 0, black],
                     [0, white, white, king,white, white, 0],
                     [black, 0, 0, white, 0, 0, black],
                     [0, 0, 0, white, 0, 0, 0],
                     [0, 0, black, black, black, 0, 0]]

    __distance_from_excapes = [[0, 0, 1, 2, 1, 0, 0],
                               [0, 2, 3, 4, 3, 2, 0],
                               [1, 3, 4, 5, 4, 3, 1],
                               [2, 4, 5, 6, 5, 4, 2],
                               [1, 3, 4, 5, 4, 3, 1],
                               [0, 2, 3, 4, 3, 2, 0],
                               [0, 0, 1, 2, 1, 0, 0]]

    camp_list = [(0, 2), (0, 3), (0, 4), (2, 0), (3, 0), (4, 0),
                 (2, 6), (3, 6), (4, 6), (6, 2), (6, 3), (6, 4)]
    escape_list = [(0, 1), (0, 5), (1, 0), (1, 6),
                   (6, 1), (6, 5), (1, 6), (6, 5)]
    __player_pieces_values = {"black": [black], "white": [white, king]}
    __weight_heuristic = {1: {"king": 10, "soldier": 1, "king_position": 30},
                          0: {"king": 5, "soldier": 10, "king_position": 20}}
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
            if position[1] >= 0 and position[1] <= 6:
                # from player pos-1 to 0 (inclusive)
                for i in reversed(range(0, position[0])):
                    # the action is ok if cell is empty, do not allow camp jump(eg. from left to right camp), go in a camp iff piece is in a camp, don't go in escape cell
                    if state[1][i][position[1]] != 0 or ((i+1, position[1]) not in self.camp_list and (i, position[1]) in self.camp_list) or ((i, position[1]) in self.camp_list and position not in self.camp_list) or (i, position[1]) == (3, 3):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], i, position[1]))
                # from player pos+1 to 8 (inclusive)
                for i in range(position[0]+1, 7):

                    if state[1][i][position[1]] != 0 or ((i-1, position[1]) not in self.camp_list and (i, position[1]) in self.camp_list) or ((i, position[1]) in self.camp_list and position not in self.camp_list) or (i, position[1]) == (3, 3):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], i, position[1]))
            # compute movement for column
            if position[0] >= 0 and position[0] <= 6:
                for i in reversed(range(0, position[1])):
                    if state[1][position[0]][i] != 0 or ((position[0], i+1) not in self.camp_list and (position[0], i) in self.camp_list) or ((position[0], i) in self.camp_list and position not in self.camp_list) or (position[0], i) == (3, 3):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], position[0], i))
                for i in range(position[1]+1, 7):
                    if state[1][position[0]][i] != 0 or ((position[0], i-1) not in self.camp_list and (position[0], i) in self.camp_list) or ((position[0], i) in self.camp_list and position not in self.camp_list) or (position[0], i) == (3, 3):
                        break
                    else:
                        actions.append(
                            (position[0], position[1], position[0], i))
        else:  # if king
            for cardinal in [(position[0]-1, position[1]), (position[0]+1, position[1]), (position[0], position[1]-1), (position[0], position[1]+1)]:
                # direction is possible if don't exit the grid and don't go in castle
                if min(cardinal) >= 0 and max(cardinal) <= 6 and state[1][cardinal[0]][cardinal[1]] == 0 and cardinal != (3, 3) and cardinal not in self.camp_list:
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
        if action[2]+2 <= 6:
            directions.append(
                [(action[2]+1, action[3]), (action[2]+2, action[3])])
        if action[3]-2 >= 0:
            directions.append(
                [(action[2], action[3]-1), (action[2], action[3]-2)])
        if action[3]+2 <= 6:
            directions.append(
                [(action[2], action[3]+1), (action[2], action[3]+2)])

        for cardinal in directions:
            # if king and in castle and sorrounded by 4 sided
            if board[cardinal[0][0]][cardinal[0][1]] == self.king and cardinal[0] == (3, 3) and board[action[2]][action[3]] == self.black and board[cardinal[0][0]-1][cardinal[0][1]] == self.black and board[cardinal[0][0]+1][cardinal[0][1]] == self.black and board[cardinal[0][0]][cardinal[0][1]-1] == self.black and board[cardinal[0][0]][cardinal[0][1]+1] == self.black:
                board[cardinal[0][0]][cardinal[0][1]] = 0
            # if king and in adjacent to the castle and sorrounded by 3 sided
            elif board[cardinal[0][0]][cardinal[0][1]] == self.king and board[action[2]][action[3]] == self.black and cardinal[0] in [(2, 3), (3, 2), (4, 3), (3, 4)] and (board[cardinal[0][0]-1][cardinal[0][1]] == self.black or (cardinal[0][0]-1, cardinal[0][1]) == (3, 3)) and (board[cardinal[0][0]+1][cardinal[0][1]] == self.black or (cardinal[0][0]+1, cardinal[0][1]) == (3, 3)) and (board[cardinal[0][0]][cardinal[0][1]-1] == self.black or (cardinal[0][0], cardinal[0][1]-1) == (3, 3)) and (board[cardinal[0][0]][cardinal[0][1]+1] == self.black or (cardinal[0][0], cardinal[0][1]+1) == (3, 3)):
                board[cardinal[0][0]][cardinal[0][1]] = 0
            # if enemy and near barrier
            elif cardinal[0] not in self.camp_list and not (board[cardinal[0][0]][cardinal[0][1]] == self.king and board[action[2]][action[3]] == self.black and cardinal[0] in [(2, 3), (3, 2), (4, 3), (3, 4), (3,3)]) and (board[cardinal[0][0]][cardinal[0][1]] != board[action[2]][action[3]] and not (board[action[2]][action[3]] == self.king and board[cardinal[0][0]][cardinal[0][1]] == self.white or board[action[2]][action[3]] == self.white and board[cardinal[0][0]][cardinal[0][1]] == self.king)) and ((board[action[2]][action[3]] == board[cardinal[1][0]][cardinal[1][1]] or (board[action[2]][action[3]] == self.white and board[cardinal[1][0]][cardinal[1][1]] == self.king or board[action[2]][action[3]] == self.king and board[cardinal[1][0]][cardinal[1][1]] == self.white)) or (cardinal[1] == (3, 3) or cardinal[1] in self.camp_list)):
                board[cardinal[0][0]][cardinal[0][1]] = 0

        return ((state[0]+1) % 2, board, state[2] + 1)

    def h(self, state: State, player: int, min_max: bool) -> float:
        num_white = len(self.where(state[1], [self.white]))
        num_black = len(self.where(state[1], [self.black]))
        king_pos = self.where(state[1], [self.king])[0]
        enemy_adjacent_king = sum([1 if state[1][around_king[0]][around_king[1]] == self.black else 0 for around_king in [(
            king_pos[0]-1, king_pos[1]), (king_pos[0]+1, king_pos[1]), (king_pos[0], king_pos[1]-1), (king_pos[0], king_pos[1]+1)]])
        player_index = -2*player+1  # 1 if player == 0 else -1
        min_max_player = 1 if (player == 0 and not min_max) or (
            player == 1 and min_max) else 0

        soldier_value = (num_white - num_black)
        king_value = (enemy_adjacent_king if king_pos == (3, 3) else enemy_adjacent_king *
                      4 if king_pos not in [(2, 3), (3, 2), (4, 3), (3, 4)] else enemy_adjacent_king*2)
        king_pos_value = self.__distance_from_excapes[king_pos[0]][king_pos[1]]

        return player_index * (self.__weight_heuristic[min_max_player]["soldier"] * soldier_value
                               - self.__weight_heuristic[min_max_player]["king"] * king_value -
                               self.__weight_heuristic[min_max_player]["king_position"] * king_pos_value)

    def is_terminal(self, state: State) -> bool:
        if state[2] >= 0:
            return True
        king_pos = self.where(state[1], [self.king])
        if len(king_pos) == 0:
            return True
        king_pos = tuple(king_pos[0])
        return True if min(*king_pos) == 0 or max(*king_pos) == 6 else False

    def utility(self, state: State, player: int) -> float:
        king_pos = self.where(state[1], [self.king])
        if len(king_pos) == 0:
            v = infinity if player == 1 else -infinity
        else:
            v = -infinity if player == 1 else infinity
        print(
            f"Final State reached: player: {player}, value: {v}, (king_pos: {king_pos})")
        return v

    def get_player_pieces_values(self, player: str) -> list[int]:
        ''' Get the type(numerical) of the pieces for a player

        Keyword arguments:
        player -- name of the player
        '''
        if player not in self.__player_pieces_values:
            raise Exception("Player not found")
        return self.__player_pieces_values[player]


if __name__ == '__main__':
    # (1, [[0, 0, 0, -1, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, -1, 0], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [-1, -1, 1, 0, 2, 1, 1, 0, -1], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, -1, -1, -1, 0, 0, 0]], -1998), (3, 7, 2, 7), [(3, 7, 3, 6), (3, 7, 3, 5)])
    # state = (1, [[0, 0, 0, -1, -1, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, -1, 0], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [-1, -1, 1, 0, 2, 1, 1, 0, -1], [-1, 0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, -1, -1, -1, 0, 0, 0]], -1998)
    # action = (3,7,2,7)

    # print(Game().actions(state))
    pass
