from games.player import Player


class Local(Player):
    ''' Class for a local player. Is based on a GUI, so if is not present this class may not work.'''

    def __init__(self, make_move, board, game, player):
        ''' Create a local player

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        '''
        super(Local, self).__init__(make_move, board, game, player)
        self.cell_highlighted = []
        self.highlighted_actions = []
        board.event.on("cell_pressed", self.cell_selected)

    def cell_selected(self, cell):
        if cell in self.cell_highlighted:
            action = [action for action in self.highlighted_actions if action[2]
                      == cell[0] and action[2] == cell[1]]
            self.cell_highlighted = []
            self.highlighted_actions = []
            self.board.select_state(self.game.result(
                self.board.state, action))
            self.make_move(action)
        elif self.board.state[1][cell[0]][cell[1]] in self.game.get_player_pieces_values(self.player):
            self.highlighted_actions = self.game.get_piece_actions(
                self.board.state, cell)
            self.board.highlight_actions(
                self.board.state, self.highlighted_actions)
            self.cell_highlighted = [(action[2], action[3])
                                     for action in self.highlighted_actions]
        else:
            self.cell_highlighted = []
            self.highlighted_actions = []
            self.board.highlight_actions(self.board.state)

    def next_action(self, last_action):
        pass
