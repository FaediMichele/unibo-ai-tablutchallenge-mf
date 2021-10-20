avaliable_game = {}


def make(game_name: str):
    if game_name in avaliable_game:
        return avaliable_game[game_name]()
    else:
        raise Exception("Environment not found")


if __name__ == '__main__':
    pass
