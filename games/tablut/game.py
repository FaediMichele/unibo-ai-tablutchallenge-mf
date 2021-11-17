import numpy as np
from games.game import Game as Gm

# example action: (start_row_id, start_column_id, dest_row_id, dest_column_id)
# example state: (player_turn, map)


class Game(Gm):
    ''' Class that contains rules for the Tablut game'''

    black = -1
    white = 1
    king = 2
    __empty_board = np.array([[0,  0,  0, black, black, black,  0,  0,  0],
                              [0,  0,  0,  0, black,  0,  0,  0,  0],
                              [0,  0,  0,  0,  white,  0,  0,  0,  0],
                              [black,  0,  0,  0,  white,  0,  0,  0, black],
                              [black, black,  white,  white,  king,
                               white,  white, black, black],
                              [black,  0,  0,  0,  white,  0,  0,  0, black],
                              [0,  0,  0,  0,  white,  0,  0,  0,  0],
                              [0,  0,  0,  0, black,  0,  0,  0,  0],
                              [0,  0,  0, black, black, black,  0,  0,  0]])
    __player_names = {"black": 1, "white": 0}
    __player_pieces_values = {"black": [black], "white": [white, king]}

    def create_root(self, player):
        return (player, self.__empty_board.copy())

    def get_piece_positions(self, state):
        ''' Returns a numpy array containing all the piece position for the current player

        Keyword arguments:
        state -- the state of the game'''
        if state[0] == self.__player_names["black"]:
            return list(zip(*np.where(state[1] == self.black)))
        else:
            return list(zip(*np.where((state[1] == self.white) | (state[1] == self.king))))

    def get_piece_actions(self, state, position):
        ''' Returns all the possible action for a specific piece.

        Keyword arguments:
        state -- the state of the board
        position -- tuple that contains row and column for a single piece
        '''
        actions = []
        for i in range(1, position[0]):
            if state[1][i, position[1]] == 0:
                actions.append((position[0], position[1], i, position[1]))
        # TODO define here the game rules
        return actions

    def actions(self, state):
        actions = []
        for d in self.get_piece_positions(state):
            actions.extend(self.get_piece_actions(state, d))
        return actions

    def result(self, state, action):
        # TODO calculate capture and new state
        state = ((state[0]+1) % 2, state[1].copy())
        state[1][action[0], action[1]], state[1][action[2], action[3]
                                                 ] = state[1][action[2], action[3]], state[1][action[0], action[1]]
        return state

    def goal_test(self, state):
        pass

    def h(self, node):
        pass

    def is_terminal(self, state):
        ''' Return true if a state determine a win or loose

        Keyword arguments:
        state -- the state that is wanted to check'''
        return True if np.count_nonzero(state[1] == self.king) == 0 else False

    def utility(self, state, player):
        ''' Return the value of this final state to player

        Keyword arguments:
        state -- the state of the game. The current player is ignored
        player -- the player that want to know the value of a final state
        '''
        return 1 if np.count_nonzero(state[1] == self.king) == 0 else -1

    def get_player_pieces_values(self, player):
        ''' Get the type(numerical) of the pieces for a player

        Keyword arguments:
        player -- name of the player
        '''
        if player not in self.__player_pieces_values:
            raise Exception("Player not found")
        return self.__player_pieces_values[player]


if __name__ == '__main__':
    pass
