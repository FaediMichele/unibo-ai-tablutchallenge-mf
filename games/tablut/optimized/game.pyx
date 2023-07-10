# Game implemented in Cython, tested using many matches with random player
#

from libcpp cimport bool
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free

cdef struct s_coordinate:
    int x
    int y

cdef struct s_action:
    int x1
    int y1
    int x2
    int y2

cdef struct s_state:
    int player
    int[9][9] board
    int remaining_moves

ctypedef s_state State
ctypedef s_coordinate Coordinate
ctypedef s_action Action


cdef Action[18] piece_actions
cdef Coordinate[16] where_coordinate
cdef Action[288] actions_board

cdef struct s_action_list:
    Action* action_ptr
    int length

cdef struct s_coordinate_list:
    Coordinate* coordinate_ptr
    int length

ctypedef s_action_list ActionList
ctypedef s_coordinate_list CoordinateList

cdef Coordinate[16] camp_list = [Coordinate(0, 3), Coordinate(0, 4), Coordinate(0, 5), Coordinate(1, 4), Coordinate(8, 3), Coordinate(8, 4), Coordinate(8, 5), Coordinate(7, 4), Coordinate(3, 0), Coordinate(4, 0), Coordinate(5, 0), Coordinate(4, 1), Coordinate(3, 8), Coordinate(4, 8), Coordinate(5, 8), Coordinate(4, 7)]
cdef Coordinate[16] escape_list = [Coordinate(0, 1), Coordinate(0, 2), Coordinate(0, 6), Coordinate(0, 7), Coordinate(8, 1), Coordinate(8, 2), Coordinate(8, 6), Coordinate(8, 7), Coordinate(1, 0), Coordinate(2, 0), Coordinate(6, 0), Coordinate(7, 0), Coordinate(1, 8), Coordinate(2, 8), Coordinate(6, 8), Coordinate(7, 8)]


cdef int get_piece_actions(State s, Coordinate pos):
    cdef int actions_index = 0
    cdef int i, j
    cdef Coordinate[4] king_positions = [Coordinate(pos.x - 1, pos.y), Coordinate(pos.x + 1, pos.y), Coordinate(pos.x, pos.y - 1), Coordinate(pos.x, pos.y + 1)]
    cdef Coordinate d
    cdef bint action_to_add, ok1
    cdef int delta, k, h


    if s.board[pos.x][pos.y] == 2: # if king
        # king_positions 
        
        for d in king_positions:
            if d.x >= 0 and d.y >= 0 and d.x <= 8 and d.y <= 8 and s.board[d.x][d.y] == 0 and not (d.x == 4 and d.y == 4):
                for j in range(16):
                    if d.x == camp_list[j].x and d.y == camp_list[j].y:
                        break
                else:
                    piece_actions[actions_index] = Action(pos.x, pos.y, d.x, d.y)
                    actions_index += 1
    elif s.board[pos.x][pos.y] == 1: # if white

        for k in range(4):
            for delta in range(1, 9):
                if k == 0:
                    if pos.x + delta >= 9:
                        break
                    d = Coordinate(pos.x + delta, pos.y)
                elif k == 1:
                    if pos.x - delta < 0:
                        break
                    d = Coordinate(pos.x - delta, pos.y)
                if k == 2:
                    if pos.y + delta >= 9:
                        break
                    d = Coordinate(pos.x, pos.y + delta)
                elif k == 3:
                    if pos.y - delta < 0:
                        break
                    d = Coordinate(pos.x, pos.y - delta)

                action_to_add = True
                if s.board[d.x][d.y] == 0 and not (d.x == 4 and d.y == 4):
                    for j in range(16):
                        if d.x == camp_list[j].x and d.y == camp_list[j].y:
                            action_to_add = False
                            break
                else:
                    action_to_add = False
                
                if action_to_add:
                    piece_actions[actions_index] = Action(pos.x, pos.y, d.x, d.y)
                    actions_index += 1
                else:
                    break
    else: # if black
        for k in range(4):
            for delta in range(1, 9):
                if k == 0:
                    if pos.x + delta >= 9:
                        break
                    d = Coordinate(pos.x + delta, pos.y)
                elif k == 1:
                    if pos.x - delta < 0:
                        break
                    d = Coordinate(pos.x - delta, pos.y)
                if k == 2:
                    if pos.y + delta >= 9:
                        break
                    d = Coordinate(pos.x, pos.y + delta)
                elif k == 3:
                    if pos.y - delta < 0:
                        break
                    d = Coordinate(pos.x, pos.y - delta)

                action_to_add = True
                if s.board[d.x][d.y] == 0 and not (d.x == 4 and d.y == 4):
                    for j in range(16):
                        if d.x == camp_list[j].x and d.y == camp_list[j].y:
                            ok1 = True
                            for h in range(16):
                                if pos.x == camp_list[h].x and pos.y == camp_list[h].y:
                                    # not camp jump (eg. from left side camp to right side camp)
                                    if not ((pos.x <= 1 and d.x >= 7) or (pos.x >= 7 and d.x <= 1) or (pos.y <= 1 and d.y >= 7) or (pos.y >= 7 and d.y <= 1)):
                                        ok1 = False
                                    break
                            if ok1:
                                action_to_add = False
                            break
                else:
                    action_to_add = False
                
                if action_to_add:
                    piece_actions[actions_index] = Action(pos.x, pos.y, d.x, d.y)
                    actions_index += 1
                else:
                    break
    return actions_index


cdef int where_black(int[9][9] board):
    cdef int h, k, index
    index = 0

    for h in range(9):
        for k in range(9):
            if board[h][k] == -1:
                where_coordinate[index] = Coordinate(h, k)
                index += 1
    return index

cdef int where_white(int[9][9] board):
    cdef int h, k, index
    index = 0

    for h in range(9):
        for k in range(9):
            if board[h][k] == 1 or board[h][k] == 2:
                where_coordinate[index] = Coordinate(h, k)
                index += 1
    return index


cdef int actions_black(State s):
    cdef int actions_for_piece_count
    cdef Action a
    cdef Coordinate p
    cdef int index = 0
    cdef int i, j

    for i in range(where_black(s.board)):
        for j in range(get_piece_actions(s, where_coordinate[i])):
            actions_board[index] = piece_actions[j]
            index += 1
    return index

cdef int actions_white(State s):
    cdef Action a
    cdef Coordinate p
    cdef int index = 0
    cdef int i, j

    for i in range(where_white(s.board)):
        for j in range(get_piece_actions(s, where_coordinate[i])):
            actions_board[index] = piece_actions[j]
            index += 1
    return index

cpdef actions(python_state):
    cdef ActionList al
    cdef int i, j
    cdef State s
    s.player = python_state[0]
    s.remaining_moves = python_state[2]

    for i in range(9):
        for j in range(9):
            s.board[i][j] = python_state[1][i][j]
    if is_terminal_c(s):
        return []
    if s.player == 0:
        j = actions_white(s)
    else:
        j = actions_black(s)

    return [(actions_board[i].x1, actions_board[i].y1, actions_board[i].x2, actions_board[i].y2) for i in range(j)]
        
cdef bint is_terminal_c(State s):
    if s.remaining_moves >= 0:
        return True
    
    for h in range(9):
        for k in range(9):
            if s.board[h][k] == 2:
                if h == 0 or k == 0 or h == 8 or k == 8:
                    return True
                return False

    return True

cpdef bint is_terminal(python_state: tuple[int, list[list[int]], int]):
    if python_state[2] >= 0:
        return True
    
    for h in range(9):
        for k in range(9):
            if python_state[1][h][k] == 2:
                if h == 0 or k == 0 or h == 8 or k == 8:
                    return True
                return False

    return True

cpdef result(python_state: tuple[int, list[list[int]], int], python_action: tuple[int,int,int,int]):
    cdef int[9][9] new_board
    cdef int[4][4] d
    cdef int i, j, h, k, cardinal_used = 0
    cdef bint king_special = False
    cdef Action a
    a.x1 = python_action[0]
    a.y1 = python_action[1]
    a.x2 = python_action[2]
    a.y2 = python_action[3]

    for i in range(9):
        for j in range(9):
            new_board[i][j] = python_state[1][i][j]
    
    new_board[a.x2][a.y2] = new_board[a.x1][a.y1]
    new_board[a.x1][a.y1] = 0

    cardinal_used = 0
    if a.x2 >= 2 and new_board[a.x2 - 1][a.y2] != 0:
        d[cardinal_used][0] = a.x2 - 1
        d[cardinal_used][1] = a.y2
        d[cardinal_used][2] = a.x2 - 2
        d[cardinal_used][3] = a.y2
        cardinal_used += 1
    if a.x2 <= 6 and new_board[a.x2 + 1][a.y2] != 0:
        d[cardinal_used][0] = a.x2 + 1
        d[cardinal_used][1] = a.y2
        d[cardinal_used][2] = a.x2 + 2
        d[cardinal_used][3] = a.y2
        cardinal_used += 1
    if a.y2 >= 2 and new_board[a.x2][a.y2 - 1] != 0:
        d[cardinal_used][0] = a.x2
        d[cardinal_used][1] = a.y2 - 1
        d[cardinal_used][2] = a.x2
        d[cardinal_used][3] = a.y2 - 2
        cardinal_used += 1
    if a.y2 <= 6 and new_board[a.x2][a.y2 + 1] != 0:
        d[cardinal_used][0] = a.x2
        d[cardinal_used][1] = a.y2 + 1
        d[cardinal_used][2] = a.x2
        d[cardinal_used][3] = a.y2 + 2
        cardinal_used += 1

    if new_board[4][4] == 2 or new_board[3][4] == 2 or new_board[4][3] == 2 or new_board[5][4] == 2 or new_board[4][5] == 2:
        king_special = True
    
    for i in range(cardinal_used):
        if new_board[a.x2][a.y2] == -1 and king_special and new_board[d[i][0]][d[i][1]] == 2:
            # if king in castle and sorrunded by 4 sides
            if d[i][0] == 4 and d[i][1] == 4 and new_board[3][4] == -1 and new_board[4][3] == -1 and new_board[5][4] == -1 and new_board[4][5] == -1:
                new_board[4][4] = 0
            # if king near castle and sorrunded by 3 sides
            elif d[i][0] == 3 and d[i][1] == 4 and new_board[3][3] == -1 and new_board[3][5] == -1 and new_board[2][4] == -1:
                new_board[3][4] = 0
            elif d[i][0] == 4 and d[i][1] == 3 and new_board[3][3] == -1 and new_board[5][3] == -1 and new_board[4][2] == -1:
                new_board[4][3] = 0
            elif d[i][0] == 4 and d[i][1] == 5 and new_board[5][5] == -1 and new_board[3][5] == -1 and new_board[4][6] == -1:
                new_board[4][5] = 0
            elif d[i][0] == 5 and d[i][1] == 4 and new_board[5][5] == -1 and new_board[5][3] == -1 and new_board[6][4] == -1:
                new_board[5][4] = 0
            
        elif new_board[a.x2][a.y2] * new_board[d[i][0]][d[i][1]] < 0: # piece next to me is an enemy
            
            for h in range(16):
                if camp_list[h].x == d[i][0] and camp_list[h].y == d[i][1]:
                    break
            else:
                if new_board[d[i][2]][d[i][3]] * new_board[a.x2][a.y2] > 0:
                    new_board[d[i][0]][d[i][1]] = 0
                for h in range(16):
                    # if enemy not in camp and (near a wall or near my ally)
                    if camp_list[h].x == d[i][2] and camp_list[h].y == d[i][3] or d[i][2] == 4 and d[i][3] == 4:
                        new_board[d[i][0]][d[i][1]] = 0
                        break
            
    return ((python_state[0] + 1) % 2, new_board, python_state[2] + 1)



