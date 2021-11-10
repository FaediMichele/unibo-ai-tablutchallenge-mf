import numpy as np
from games.game import Game as Gm

# example action: (start_row_id, start_column_id, dest_row_id, dest_column_id)
# example state: (player_turn, map)


class Game(Gm):
    ''' Class that contains rules for the Tablut game'''

    __black = -1
    __white = 1
    __king = 2
    __empty_board = np.array([[0,  0,  0, __black, __black, __black,  0,  0,  0],
                              [0,  0,  0,  0, __black,  0,  0,  0,  0],
                              [0,  0,  0,  0,  __white,  0,  0,  0,  0],
                              [__black,  0,  0,  0,  __white,  0,  0,  0, __black],
                              [__black, __black,  __white,  __white,  __king,
                               __white,  __white, __black, __black],
                              [__black,  0,  0,  0,  __white,  0,  0,  0, __black],
                              [0,  0,  0,  0,  __white,  0,  0,  0,  0],
                              [0,  0,  0,  0, __black,  0,  0,  0,  0],
                              [0,  0,  0, __black, __black, __black,  0,  0,  0]])
    __player_names = {"black": __black, "white": __white}
    __player_pieces_values = {"black": [__black], "white": [__white, __king]}

    def create_root(self, player):
        return (self.__player_names[player], self.__empty_board.copy())

    def get_piece_positions(self, state):
        ''' Returns a numpy array containing all the piece position for the current player

        Keyword arguments:
        state -- the state of the game'''
        if state[0] == "black":
            return zip(np.where(state[1] == self.__black))
        else:
            return zip(np.where((state[1] == self.__white) & (state[1] == self.__king)))

    def get_piece_actions(self, state, position):
        ''' Returns all the possible action for a specific piece.

        Keyword arguments:
        state -- the state of the board
        position -- tuple that contains row and column for a single piece
        '''
        # TODO define here the game rules
        return []

    def actions(self, state):
        actions = []
        for d in self.get_piece_positions(state):
            actions.append(self.get_piece_actions(state, d))
        return actions

    def result(self, state, action):
        # TODO calculate capture and new state
        return state

    def goal_test(self, state):
        pass

    def h(self, node):
        pass

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
