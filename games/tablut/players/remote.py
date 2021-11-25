from games.player import Player
import json
import asyncio
import string
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

    async def connect(self):
        ''' Open the connection to the server '''
        self.reader, self.writer = await asyncio.open_connection(*self.address)
        logging.info(f'Connected, {self.reader}, {self.writer}')

    async def close(self):
        ''' Close the connection'''
        self.writer.close()
        await self.writer.wait_closed()

    async def send(self, data):
        ''' Send data and wait for the response.

        Keyword arguments:
        data -- data to send
        '''
        logging.info(f'sending: {data}')
        self.writer.write(msg_prepare(data))
        await self.writer.drain()

    async def read(self):
        ret = await self.reader.read(self.buffer_size)
        # The integer size seem to be sent separately (idk really)
        # so it will sometimes be received without his message.
        # In that case, wait for the message and unpack the whole thing.
        if len(ret) == 4:
            ret += await self.reader.read(self.buffer_size)
        return msg_unpack(ret)


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

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.enemy.connect())
        # asyncio.run(self.enemy.connect())
        self._send_name()
        self.map_names = {"EMPTY": 0, "BLACK": game.black,
                          "WHITE": game.white, "KING": game.king}

    async def _send_name_async(self):
        """Handshake with the server sending the player's name."""
        await self.enemy.send(f'\"{self.name}\"')

        # Consume first configuration
        await self.enemy.read()

    def _send_name(self):
        """Wrapper on `_send_name`."""
        asyncio.get_event_loop().run_until_complete(self._send_name_async())
        # asyncio.run(self._send_name())

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

    async def next_action_async(self, last_action):
        # First turn, don't send anything and wait for the server's
        # first state
        if last_action is not None:
            # Send and consume server response (an echo of the new state)
            await self.enemy.send(self.encode(last_action))
            self.decode(await self.enemy.read())

        # Wait for remote action
        self.make_move(self.decode(await self.enemy.read()))

    def next_action(self, last_action):
        asyncio.get_event_loop().run_until_complete(self.next_action_async(last_action))
        # asyncio.run(self.next_action_async(last_action))
