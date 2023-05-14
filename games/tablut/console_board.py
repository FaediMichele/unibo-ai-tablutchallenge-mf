from games.board import Board, zeros_matrix
from termcolor import colored, COLORS
from copy import deepcopy

empty_colored_board =[
    [
        colored('■', 'cyan'),
        colored('■', 'cyan'),
        colored('■', 'cyan'),
        colored('■', 'light_blue'),
        colored('■', 'light_blue'),
        colored('■', 'light_blue'),
        colored('■', 'cyan'),
        colored('■', 'cyan'),
        colored('■', 'cyan'),
    ],
    [
        colored('■', 'cyan'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_blue'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'cyan'),
    ],
    [
        colored('■', 'cyan'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'cyan'),
    ],
    [
        colored('■', 'light_blue'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_blue'),
    ],
    [
        colored('■', 'light_blue'),
        colored('■', 'light_blue'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'magenta'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_blue'),
        colored('■', 'light_blue'),
    ],
    [
        colored('■', 'light_blue'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_blue'),
    ],
    [
        colored('■', 'cyan'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'cyan'),
    ],
    [
        colored('■', 'cyan'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_blue'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'light_yellow'),
        colored('■', 'cyan'),
    ],
    [
        colored('■', 'cyan'),
        colored('■', 'cyan'),
        colored('■', 'cyan'),
        colored('■', 'light_blue'),
        colored('■', 'light_blue'),
        colored('■', 'light_blue'),
        colored('■', 'cyan'),
        colored('■', 'cyan'),
        colored('■', 'cyan'),
    ]
]



class ConsoleBoard(Board):
    def __init__(self, initial_state=zeros_matrix((9,9))):
        super().__init__(initial_state)
        self.white_player = initial_state[0]

    def select_state(self, state):
        res = super().select_state(state)
        self.print_board()
        return res

    def print_board(self):
        value_char = {-1: colored('B', 'white'),
                      1: colored('W', 'white'),
                      2: colored('K', 'red')}
        print()
        print(f"Turn of player {('B' if self.state[0] == 1 else 'W')} - "
              f"Moves till the end of the match {-self.state[2]}")
        print("    0 1 2 3 4 5 6 7 8\n")
        new_board = deepcopy(empty_colored_board)
        for x in range(9):
            for y in range(9):
                if self.state[1][x][y] != 0:
                    new_board[x][y] = value_char[self.state[1][x][y]]
        
        for y in range(9):
            print(chr(ord("a") + y) + "   " + " ".join([new_board[x][y]
                                                         for x in range(9)]))
            