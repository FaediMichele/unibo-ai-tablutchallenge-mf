import asyncio
from games.board import Board as Bd, zeros_matrix
from kivymd.app import MDApp
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from kivy.core.window import Window
from pymitter import EventEmitter
from kivy.clock import Clock
import os
package_directory = os.path.dirname(os.path.abspath(__file__))


class MyButton(ButtonBehavior, Image):
    def __init__(self, **kwargs):

        if "off" in kwargs:
            self.source_off = kwargs.pop('off')
        else:
            self.source_off = 'atlas://data/images/defaulttheme/checkbox_off'
        if "on" in kwargs:
            self.source_on = kwargs.pop('on')
        else:
            self.source_on = 'atlas://data/images/defaulttheme/checkbox_on'
        self.source = self.source_off
        print(kwargs)
        super(MyButton, self).__init__(**kwargs)

    def on_press(self):
        self.source = self.source_on

    def on_release(self):
        self.source = self.source_off

    def update_source(self, on=None, off=None):
        on = on or self.source_on
        off = off or self.source_off
        if self.source == self.source_off:
            self.source_off = off
            self.source_on = on
            self.source = off
        else:
            self.source_off = off
            self.source_on = on
            self.source = on


class Board(Bd, MDApp):
    ''' Class that contains rules for the Tablut game with user interfaces.
    If the Gui is not used is suggested to use the base implementation of board(games.board.Board)'''

    def __init__(self, initial_state=zeros_matrix((9, 9)), icons={
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
            "highlight": os.path.join(package_directory, 'res', "cell_10.png"),
            "white_escapes": os.path.join(package_directory, 'res', "cell_11.png"),
            "black_escapes": os.path.join(package_directory, 'res', "cell_12.png")
    }):
        super(Board, self).__init__(initial_state)
        self.icons = icons
        Window.bind(on_resize=self.on_window_resize)

    def on_window_resize(self, window, width, height):
        min_size = min(width, height)
        self.grid.size = (min_size, min_size)

    def build(self):
        min_size = min(*Window.size)
        anchor = AnchorLayout(anchor_x='center', anchor_y='center')
        self.grid = GridLayout(cols=9, size=(
            min_size, min_size), size_hint=(None, None))
        self.cells = []
        for _ in range(9):
            self.cells.append([])
            for _ in range(9):
                btn = MyButton(allow_stretch=True, on=self.icons["highlight"])
                btn.size_hint = (1, 1)
                btn.bind(on_press=self.on_btn_pressed)
                self.cells[-1].append(btn)
                self.grid.add_widget(self.cells[-1][-1])

        def on_ready(_):
            self.select_state(self.state)
            self.event.emit("loaded")
        Clock.schedule_once(on_ready, 0)
        anchor.add_widget(self.grid)
        return anchor

    def on_btn_pressed(self, val):
        for i in range(len(self.cells)):
            for k in range(len(self.cells[i])):
                if val == self.cells[i][k]:
                    self.event.emit("cell_pressed", (i, k))
                    return

    def get_img_for_cell(self, i, k, val):
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
            if val == 2:
                return self.icons["king_cell"]
            if val > 0:
                return self.icons["white_escapes"]
            if val < 0:
                return self.icons["black_escapes"]
            return self.icons["empty_escapes"]
        if (i, k) in [(0, 0), (8, 0), (0, 8), (8, 8)]:
            if val >= 1:
                return self.icons["white_cell"]
            if val < 0:
                return self.icons["black_cell"]
            if val == 0:
                return self.icons["empty_cell"]

        raise Exception(f"Cell state not found ({i}, {k}, {val})")

    def select_state(self, state):
        self.state = state
        for i in range(len(self.cells)):
            for k in range(len(self.cells[i])):
                self.cells[i][k].update_source(off=self.get_img_for_cell(
                    i, k, state[1][i][k]))

    def highlight_actions(self, state, actions=[]):
        print("hightlighted")
        self.select_state(state)
        for a in actions:
            self.cells[a[2]][a[3]].update_source(off=self.icons["highlight"])

    def run(self):
        MDApp.run(self)

    def add_manager_function(self, function):
        def f(_):
            Clock.tick()
            co = function()
            if asyncio.iscoroutinefunction(co):
                asyncio.run(co)

        self.manager_function = f

    def run_manager_function(self):
        print("Add schedule")
        Clock.schedule_once(self.manager_function, 0.1)


if __name__ == '__main__':
    Board().run()
