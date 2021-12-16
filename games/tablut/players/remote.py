from games.player import Player
import json
import string
import socket
import logging


# Dictionary used to encode the current turn in the format
# wanted by the server
TURN_ECONDING = {0: 'W', 1: 'B', "WHITE": 0, "BLACK": 1}


def msg_prepare(data: str):
    """Return a prepared sequence of bytes for sending."""
    bytes_data = data.encode()
    return len(bytes_data).to_bytes(4, byteorder='big') + bytes_data


def msg_unpack(data: bytes):
    # TODO check this function
    size = int.from_bytes(data[:4], byteorder='big')
    logging.info(f'Reading size: {data[:4]} = {size}')
    buf = data[4: 4 + size]
    logging.info(f"Read decode: {buf}")
    return buf.decode()


class Client():
    ''' Class the encapsulate the send and receive of commands for the remote player/server.
    Simple use case:
    client = Client(("127.0.0.1", 8080))
    await client.connect()
    print(f"response: {await cliend.send('hello server')}")
    await client.close()
    '''

    def __init__(self, address, buffer_size=1024):
        ''' Create a new Client TCP

        Keyword arguments:
        address -- tuple that contains address, port. For more information see this https://docs.python.org/3/library/asyncio-stream.html#asyncio.open_connection
        '''
        super(Client, self).__init__()
        self.address = address
        self.buffer_size = buffer_size
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        ''' Open the connection to the server '''
        self.socket.connect(self.address)
        logging.info(f'Connected, {self.socket}')

    def close(self):
        ''' Close the connection'''
        self.socket.close()

    def send(self, data):
        ''' Send data and wait for the response.

        Keyword arguments:
        data -- data to send
        '''
        logging.info(f'sending: {data}')
        self.socket.sendall(msg_prepare(data))

    def read(self):
        """Read data from the socket.

        Data is assumed to be prepended with a 4 bytes integer
        representing the size of the message.

        In order to avoid strange behaviours of the competition's
        network, the message will be read byte per byte.
        """
        # Read 4 bytes
        recv = bytes()
        while len(recv) < 4:
            recv += self.socket.recv(1)

        # Read all the message
        size = int.from_bytes(recv[:4], byteorder='big')
        while len(recv) - 4 < size:
            recv += self.socket.recv(1)
        return msg_unpack(recv)


class Remote(Player):
    ''' Class for a remote player '''

    def __init__(self, make_move, board, game, player, enemy_address, name=None):
        ''' Create a remote player.

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        enemy_address -- the data for comunicatin with the remote player ( #TODO yet to be defined )
        '''

        super(Remote, self).__init__(make_move, board, game, player)
        self.enemy = Client(enemy_address)

        self.name = name or str(player)

        self.enemy.connect()
        self._send_name()
        self.map_names = {"EMPTY": 0, "BLACK": game.black,
                          "WHITE": game.white, "KING": game.king,
                          "THRONE": 0}

    def _send_name(self):
        """Handshake with the server sending the player's name."""
        self.enemy.send(f'\"{self.name}\"')

        # Consume first configuration
        self.enemy.read()

    def encode(self, action):
        ''' Parse an action to the server comunication format'''
        # {"from":"e4","to":"e5","turn":"WHITE"}
        return json.dumps(
            {
                "from": (string.ascii_lowercase[action[1]] + str(action[0]+1)),
                "to": (string.ascii_lowercase[action[3]] + str(action[2]+1)),
                "turn": TURN_ECONDING.get(self.board.state[0], 'W')
            })

    def decode(self, result):
        ''' Parse a server comunication format to an action'''
        print(f"Received data: {result}")
        data = json.loads(result)
        new_state = []
        received_state = data["board"]
        for i in range(len(received_state)):
            line = []
            for j in range(len(received_state[i])):
                line.append(self.map_names[received_state[i][j]])
            new_state.append(line)
        print(f"Calculated state: {new_state}")
        print("IM HERE", (TURN_ECONDING[data["turn"]], new_state))
        return (TURN_ECONDING[data["turn"]], new_state)

    def next_action(self, last_action):
        # First turn, don't send anything and wait for the server's
        # first state
        if last_action is not None:
            # Send and consume server response (an echo of the new state)
            self.enemy.send(self.encode(last_action))
            self.decode(self.enemy.read())

        # Wait for remote action
        self.make_move(self.decode(self.enemy.read()))

    def end(self, last_action, winning):
        """Notify the winning move to the server."""
        logging.info('Sending winning move')
        self.enemy.send(self.encode(last_action))
