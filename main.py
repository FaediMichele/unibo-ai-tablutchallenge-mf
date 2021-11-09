import tablut.board as tab
import tablut.game as gm

if __name__ == '__main__':
    board = tab.Board(initial_state=gm.create_root())

    def cell_pressed(data):
        print(f"cell pressed: {data}")

    def loaded():
        print("Ready")
    board.event.on("cell_pressed", cell_pressed)
    board.event.on("loaded", loaded)

    board.run()
