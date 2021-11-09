class Player:
    def __init__(self, make_move):
        self.make_move = make_move
        super(Player, self).__init__(self)

    def next_action(self, state):
        pass
