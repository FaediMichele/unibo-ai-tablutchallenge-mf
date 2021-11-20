from games.player import Player


class Kivy(Player):
    ''' Class for a player using the kivy interface. In order to use this player the board must be games.tablut.board'''

    def __init__(self, make_move, board, game, player):
        ''' Create a local player

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        '''
        super(Kivy, self).__init__(make_move, board, game, player)
        self.cell_highlighted = []
        self.highlighted_actions = []
        self.my_turn = False
        board.event.on("cell_pressed", self.cell_selected)

    def cell_selected(self, cell):
        if not self.my_turn:
            print("not my turn")
            return
        if cell in self.cell_highlighted:
            action = [action for action in self.highlighted_actions if action[2]
                      == cell[0] and action[3] == cell[1]]
            self.cell_highlighted = []
            self.highlighted_actions = []

            self.my_turn = False
            if len(action) == 0:
                print(f"Draw. The {self.player} can not do any moves")
                self.make_move(None)
            else:
                self.make_move(action[0])

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
        print(f"Player {self.player} turn. Last action: {last_action}")
        self.my_turn = True
