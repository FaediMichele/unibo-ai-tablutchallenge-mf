from games.player import Player, State


class Console(Player):
    ''' Class for a player using the kivy interface. In order to use this player the board must be games.tablut.board'''

    def __init__(self, make_move, board, game, player):
        ''' Create a local player

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        '''
        super(Console, self).__init__(make_move, board, game, player)

    def next_action(self, last_action, state_history: list[State]):
        while True:
            string = input(
                f"Player {self.player} turn. Last action: {last_action}. Your move: ")
            if len(string) == 5:
                try:
                    action = (int(string[1]), ord(string[0])-ord("a"),
                              int(string[4]), ord(string[3])-ord("a"))
                    if min(action) >= 0 and max(action) <= 8:
                        self.make_move(action)
                        return
                    else:
                        print(action)
                except Exception as e:
                    print(e)
                    pass
            print("Wrong action format. Example format: a2,c2")
