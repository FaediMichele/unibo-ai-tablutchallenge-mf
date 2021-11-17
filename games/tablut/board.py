from games.board import Board as Bd
from kivymd.app import MDApp
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from pymitter import EventEmitter
import numpy as np
from kivy.clock import Clock
import os
package_directory = os.path.dirname(os.path.abspath(__file__))


class Board(Bd, MDApp):
    ''' Class that contains rules for the Tablut game with user interfaces.
    If the Gui is not used is suggested to use the base implementation of board(games.board.Board)'''

    def __init__(self, initial_state=np.zeros((9, 9), dtype=np.int32), icons={
            "empty_cell": os.path.join(package_directory, 'res', "cell_0.png"),
            "empty_cell_1": os.path.join(package_directory, 'res', "cell_3.png"),
            "empty_camp": os.path.join(package_directory, 'res', "cell_2.png"),
            "empty_castle": os.path.join(package_directory, 'res', "cell_4.png"),
            "empty_escapes": os.path.join(package_directory, 'res', "cell_1.png"),
            "king_castle": os.path.join(package_directory, 'res', "cell_6.png"),
            "black_cell": os.path.join(package_directory, 'res', "cell_7.png"),
            "white_cell": os.path.join(package_directory, 'res', "cell_8.png"),
            "black_camp": os.path.join(package_directory, 'res', "cell_5.png"),
            "king_cell": os.path.join(package_directory, 'res', "cell_9.png"),
            "highlight": os.path.join(package_directory, 'res', "cell_10.png")
    }):
        super(Board, self).__init__(initial_state)
        self.icons = icons

    def build(self):
        grid = GridLayout(cols=9)
        self.cells = []
        for _ in range(9):
            self.cells.append([])
            for _ in range(9):
                btn = Button()
                btn.bind(on_press=self.on_btn_pressed)
                self.cells[-1].append(btn)
                grid.add_widget(self.cells[-1][-1])

        def on_ready(_):
            self.select_state(self.state)
            self.event.emit("loaded")
        Clock.schedule_once(on_ready, 0)
        return grid

    def on_btn_pressed(self, val):
        for i in range(len(self.cells)):
            for k in range(len(self.cells[i])):
                if val == self.cells[i][k]:
                    self.event.emit("cell_pressed", (i, k))
                    return

    def get_img_for_cell(self, i, k, val):
        if (i == 0 or i == 8) and (k == 0 or k == 8):
            return self.icons["empty_cell"]
        if (i >= 1 and i <= 3 or i >= 5 and i <= 7) and (k >= 1 and k <= 3 or k >= 5 and k <= 7):
            if val < 0:
                return self.icons["black_cell"]
            if val == 0:
                return self.icons["empty_cell"]
            if val == 1:
                return self.icons["white_cell"]
            if val > 1:
                return self.icons["king_cell"]
        if (i == 4 and k >= 2 and k <= 6 and k != 4) or (k == 4 and i >= 2 and i <= 6 and i != 4):
            if val < 0:
                return self.icons["black_cell"]
            if val == 0:
                return self.icons["empty_cell_1"]
            if val == 1:
                return self.icons["white_cell"]
            if val > 1:
                return self.icons["king_cell"]
        if i == 4 and k == 4:
            if val > 1:
                return self.icons["king_castle"]
            else:
                return self.icons["empty_castle"]
        if i == 4 and (k <= 1 or k >= 7) or k == 4 and (i <= 1 or i >= 7) or (i == 0 or i == 8) and k >= 3 and k <= 5 or (k == 0 or k == 8) and i >= 3 and i <= 5:
            if val < 0:
                return self.icons["black_camp"]
            else:
                return self.icons["empty_camp"]
        if (i == 0 or i == 8) and (k >= 1 and k <= 2 or k >= 5 and k <= 7) or (k == 0 or k == 8) and (i >= 1 and i <= 2 or i >= 5 and i <= 7):
            if val > 1:
                return self.icons["king_cell"]
            return self.icons["empty_escapes"]

        raise Exception(f"Cell state not found ({i}, {k}, {val})")

    def select_state(self, state):
        self.state = state
        for i in range(len(self.cells)):
            for k in range(len(self.cells[i])):
                self.cells[i][k].background_normal = self.get_img_for_cell(
                    i, k, state[1][i, k])

    def highlight_actions(self, state, actions=[]):
        print("hightlighted")
        self.select_state(state)
        for a in actions:
            self.cells[a[2]][a[3]].background_normal = self.icons["highlight"]

    def run(self):
        MDApp.run(self)


if __name__ == '__main__':
    Board().run()
