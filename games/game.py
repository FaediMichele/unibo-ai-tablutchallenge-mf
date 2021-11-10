class Game:
    ''' Abstract class that represent the game rule.
    Does not contain the state but the method that allow to create the decision tree starting from a state
    '''

    def create_root():
        ''' Return the initial state of the game '''
        pass

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        pass

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        pass

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return any(x is state for x in self.goal)
        else:
            return state == self.goal

    def h(self, node):
        ''' Euristic function'''
        pass

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def get_player_pieces_values(self, state):
        '''This function return all the type of pieces that a player have in game'''
        pass
