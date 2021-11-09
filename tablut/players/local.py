from player import Player
from game import get_piece_actions, calculate_next_state


class Local(Player):
    def __init__(self, make_move, board, player):
        super(Local, self).__init__(self, make_move)
        self.board = board
        self.player = player
        self.cell_highlighted = []
        self.highlighted_actions = []
        board.events.on("cell_pressed", self.cell_selected)

    def cell_selected(self, cell):
        if cell in self.cell_highlighted:
            action = [action for action in self.highlighted_actions if action[2]
                      == cell[0] and action[2] == cell[1]]
            self.cell_highlighted = []
            self.highlighted_actions = []
            self.board.select_state(calculate_next_state(
                self.board.state, action))
            self.make_move(action)
            return
        if self.board.state[cell[0]][cell[1]] == self.player:
            self.highlighted_actions = get_piece_actions(cell, self.player)
            self.board.highlight_actions(self.state, self.highlighted_actions)
            self.cell_highlighted = [(action[2], action[3])
                                     for action in self.highlighted_actions]

    def next_action(self, state, last_action):
        pass
