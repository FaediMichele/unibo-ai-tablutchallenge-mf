from games.board import Board
from termcolor import colored, COLORS
from copy import deepcopy

empty_colored_board =[
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ],
    [
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
    ]
]

def zeros_matrix(shape):
    return [[0 for _ in range(shape[1])] for _ in range(shape[0])]

class ConsoleBoard(Board):
    def __init__(self, initial_state=zeros_matrix((9,9))):
        super().__init__(initial_state)
        self.white_player = initial_state[0]

    def select_state(self, state):
        res = super().select_state(state)
        self.print_board()
        return res

    def print_board(self):
        value_char = {-1: colored('X', 'white'),
                      1: colored('O', 'white')}
        print()
        print(f"Turn of player {('X' if self.state[0] == 1 else 'O')} - {self.state[0]}")
        print("    0 1 2 3 4 5 6 7 8\n")
        new_board = deepcopy(empty_colored_board)
        for x in range(9):
            for y in range(9):
                if self.state[1][x][y] != 0:
                    new_board[x][y] = value_char[self.state[1][x][y]]
        
        for y in range(9):
            print(chr(ord("a") + y) + "   " + " ".join([new_board[x][y]
                                                         for x in range(9)]))
            