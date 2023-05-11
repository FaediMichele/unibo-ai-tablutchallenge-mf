from games.tablut.players.alpha_tablut_zero import AlphaTablutZero, ModelUtil, model


def main():
    player = AlphaTablutZero(None, None, None, 0)
    ModelUtil.train_model(model, None)

if __name__ == '__main__':
    main()


