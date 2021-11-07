from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.utils import get_color_from_hex
import kivy.metrics as metrics
from pymitter import EventEmitter
import numpy as np
import game


class Board(MDApp):
    def __init__(self, icons={"empty_cell": "res/cell_0.png", "empty_cell_1": "res/cell_3.png", "empty_camp": "res/cell_2.png", "empty_castle": "res/cell_4.png", "empty_escapes": "res/cell_1.png", "king_castle": "res/cell_6.png", "black_cell": "res/cell_7.png", "white_cell": "res/cell_8.png", "black_camp": "res/cell_5.png", "king_cell": "res/cell_9.png", "highlight": "res/cell_10.png"}):
        super().__init__()
        self.event = EventEmitter()
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
            return self.icons["empty_escapes"]

        raise Exception(f"Cell state not found ({i}, {k}, {val})")

    def select_state(self, state):
        for i in range(len(self.cells)):
            for k in range(len(self.cells[i])):
                self.cells[i][k].background_normal = self.get_img_for_cell(
                    i, k, state[i, k])

    def highlight_actions(self, state, actions=[]):
        self.select_state(state)
        for a in actions:
            self.cells[a[2]][a[3]].background_normal = self.icons["highlight"]


if __name__ == '__main__':
    Board().run()
