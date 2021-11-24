from games.player import Player
import json
import asyncio
import string
import logging


# Dictionary used to encode the current turn in the format
# wanted by the server
TURN_ECONDING = {0: 'W', 1: 'B'}


def msg_prepare(data: str):
    """Return a prepared sequence of bytes for sending."""
    bytes_data = data.encode()
    return len(bytes_data).to_bytes(4, byteorder='big') + bytes_data


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

    async def send(self, data, response=True):
        ''' Send data and wait for the response.

        Keyword arguments:
        data -- data to send
        response -- wether we should wait for a response or not
        '''
        logging.info(f'sending: {data}')
        self.writer.write(msg_prepare(data))
        await self.writer.drain()

        if response:
            return await self.reader.read(self.buffer_size)


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
        self._send_name_async()

    async def _send_name(self):
        """Handshake with the server sending the player's name."""
        await self.enemy.send(f'\"{self.name}\"', response=False)

    def _send_name_async(self):
        """Wrapper on `_send_name`."""
        asyncio.get_event_loop().run_until_complete(self._send_name())
        # asyncio.run(self._send_name())

    def encode(self, action):
        ''' Parse an action to the server comunication format'''
        # {"from":"e4","to":"e5","turn":"WHITE"}
        return json.dumps(
            {
                "from": (string.ascii_lowercase[action[0]] + str(action[1])),
                "to": (string.ascii_lowercase[action[2]] + str(action[3])),
                "turn": TURN_ECONDING.get(self.board.state[0], 'W')
            })

    def decode(self, result):
        ''' Parse a server comunication format to an action'''
        data = json.load(result)
        return (ord(data["from"][0]) - ord("a"), int(data["from"][1]), ord(data["to"][0]) - ord("a"), int(data["to"][1]))

    async def next_action_async(self, last_action):
        self.make_move(self.decode(await self.enemy.send(self.encode(last_action))))

    def next_action(self, last_action):
        asyncio.get_event_loop().run_until_complete(self.next_action_async(last_action))
        # asyncio.run(self.next_action_async(last_action))
