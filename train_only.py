from games.tablut.players.alphazero.alpha_tablut_zero import AlphaTablutZero, ModelUtil


def main():
    model = ModelUtil.load_model()
    ModelUtil.train_model(model, epochs=20)
    ModelUtil.save_model(model)


if __name__ == '__main__':
    main()


