from main import main
from games.tablut.players.minmax import MinMax
from games.tablut.kiyy_board import Board as KivyBoard
from games.tablut.players.kivy import Kivy

if __name__ == '__main__':
    main([('white', Kivy, tuple()), ('black', MinMax, tuple(10))], boardtype=KivyBoard)
