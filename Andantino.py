# Generated code -- CC0 -- No Rights Reserved -- http://www.redblobgames.com/grids/hexagons/


import collections
import math
import copy 
import pygame
import pygame.gfxdraw


Point = collections.namedtuple("Point", ["x", "y"])
_Hex = collections.namedtuple("Hex", ["q", "r", "s"])

def Hex(q, r, s):
    assert not (round(q + r + s) != 0), "q + r + s must be 0"
    return _Hex(q, r, s)

def hex_add(a, b):
    return Hex(a.q + b.q, a.r + b.r, a.s + b.s)

def hex_subtract(a, b):
    return Hex(a.q - b.q, a.r - b.r, a.s - b.s)

def hex_scale(a, k):
    return Hex(a.q * k, a.r * k, a.s * k)

def hex_rotate_left(a):
    return Hex(-a.s, -a.q, -a.r)

def hex_rotate_right(a):
    return Hex(-a.r, -a.s, -a.q)

hex_directions = [Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)]
def hex_direction(direction):
    return hex_directions[direction]

def hex_neighbor(hex, direction):
    return hex_add(hex, hex_direction(direction))

hex_diagonals = [Hex(2, -1, -1), Hex(1, -2, 1), Hex(-1, -1, 2), Hex(-2, 1, 1), Hex(-1, 2, -1), Hex(1, 1, -2)]
def hex_diagonal_neighbor(hex, direction):
    return hex_add(hex, hex_diagonals[direction])

def hex_length(hex):
    return (abs(hex.q) + abs(hex.r) + abs(hex.s)) // 2

def hex_distance(a, b):
    return hex_length(hex_subtract(a, b))

def hex_round(h):
    qi = int(round(h.q))
    ri = int(round(h.r))
    si = int(round(h.s))
    q_diff = abs(qi - h.q)
    r_diff = abs(ri - h.r)
    s_diff = abs(si - h.s)
    if q_diff > r_diff and q_diff > s_diff:
        qi = -ri - si
    else:
        if r_diff > s_diff:
            ri = -qi - si
        else:
            si = -qi - ri
    return Hex(qi, ri, si)

def hex_lerp(a, b, t):
    return Hex(a.q * (1.0 - t) + b.q * t, a.r * (1.0 - t) + b.r * t, a.s * (1.0 - t) + b.s * t)

def hex_linedraw(a, b):
    N = hex_distance(a, b)
    a_nudge = Hex(a.q + 0.000001, a.r + 0.000001, a.s - 0.000002)
    b_nudge = Hex(b.q + 0.000001, b.r + 0.000001, b.s - 0.000002)
    results = []
    step = 1.0 / max(N, 1)
    for i in range(0, N + 1):
        results.append(hex_round(hex_lerp(a_nudge, b_nudge, step * i)))
    return results




OffsetCoord = collections.namedtuple("OffsetCoord", ["col", "row"])

EVEN = 1
ODD = -1
def qoffset_from_cube(offset, h):
    col = h.q
    row = h.r + (h.q + offset * (h.q & 1)) // 2
    return OffsetCoord(col, row)

def qoffset_to_cube(offset, h):
    q = h.col
    r = h.row - (h.col + offset * (h.col & 1)) // 2
    s = -q - r
    return Hex(q, r, s)

def roffset_from_cube(offset, h):
    col = h.q + (h.r + offset * (h.r & 1)) // 2
    row = h.r
    return OffsetCoord(col, row)

def roffset_to_cube(offset, h):
    q = h.col - (h.row + offset * (h.row & 1)) // 2
    r = h.row
    s = -q - r
    return Hex(q, r, s)




DoubledCoord = collections.namedtuple("DoubledCoord", ["col", "row"])

def qdoubled_from_cube(h):
    col = h.q
    row = 2 * h.r + h.q
    return DoubledCoord(col, row)

def qdoubled_to_cube(h):
    q = h.col
    r = (h.row - h.col) // 2
    s = -q - r
    return Hex(q, r, s)

def rdoubled_from_cube(h):
    col = 2 * h.q + h.r
    row = h.r
    return DoubledCoord(col, row)

def rdoubled_to_cube(h):
    q = (h.col - h.row) // 2
    r = h.row
    s = -q - r
    return Hex(q, r, s)



Orientation = collections.namedtuple("Orientation", ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"])
Layout = collections.namedtuple("Layout", ["orientation", "size", "origin"])
layout_pointy = Orientation(math.sqrt(3.0), math.sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0, math.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5)
layout_flat = Orientation(3.0 / 2.0, 0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0), 2.0 / 3.0, 0.0, -1.0 / 3.0, math.sqrt(3.0) / 3.0, 0.0)

def hex_to_pixel(layout, h):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    x = (M.f0 * h.q + M.f1 * h.r) * size.x
    y = (M.f2 * h.q + M.f3 * h.r) * size.y
    return Point(x + origin.x, y + origin.y)

def pixel_to_hex(layout, p):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
    q = M.b0 * pt.x + M.b1 * pt.y
    r = M.b2 * pt.x + M.b3 * pt.y
    return Hex(q, r, -q - r)

def hex_corner_offset(layout, corner):
    M = layout.orientation
    size = layout.size
    angle = 2.0 * math.pi * (M.start_angle - corner) / 6.0
    return Point(size.x * math.cos(angle), size.y * math.sin(angle))

def polygon_corners(layout, h):
    corners = []
    center = hex_to_pixel(layout, h)
    for i in range(0, 6):
        offset = hex_corner_offset(layout, i)
        corners.append(Point(center.x + offset.x, center.y + offset.y))
    return corners

def find_legal_move(steps,coordinates,coordmap):#Checked
    valid_moves=[]
    for i in steps:
#        print('step i: ',i)
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        neighbors=[]
    
        for y in range(len(hex_directions)): #Check all six neighbors of this step
            neighbors.append(hex_neighbor(hex_value,y))
#        print(neighbors)
        for z in neighbors:
#            print('z',z)
            if coordmap[coordinates.index(list(z))]==0:
                surround_nr=0
                for n in range(len(neighbors)):
                    if coordmap[coordinates.index(list(hex_neighbor(z,n)))]!=0:
                        surround_nr +=1
#                    print('surround_nr',surround_nr)
                    if surround_nr>1 and z not in valid_moves:
                        valid_moves.append(z)
            else:
                continue
    
    return valid_moves


def check_win_fiveinarow(is_player1_win, is_player2_win, steps, coordinates, coordmap): #Checked
    spots_player1=steps[::2]
    spots_player2=steps[1::2]
                
    hex_spots_player1=[]
    hex_spots_player2=[]
    
    for i in spots_player1:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player1.append(hex_value)
        
    for i in spots_player2:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player2.append(hex_value)    
        
    sorted_hex_spots_player1=sorted(hex_spots_player1) #Sorted in minq, minr, mins
    sorted_hex_spots_player2=sorted(hex_spots_player2)
    
#    print('sorted_hex_spots_player1', sorted_hex_spots_player1)  #Checked
#    print('sorted_hex_spots_player2', sorted_hex_spots_player2)  #Checked

    for spot in sorted_hex_spots_player1:
#        print('spot is:', spot)
        
        for d in range(6):
#            print('direction is ',d)
            for i in range(5):
                cnt=0
                i=0
                while i<5:
                    next_spot=hex_neighbor(spot,d)
#                    print('i',i)
#                    print('next_spot',next_spot)
                    if next_spot in sorted_hex_spots_player1:
                        spot=next_spot
                        i+=1
                        cnt+=1
                        if cnt==4:
                            is_player1_win=True
                            break
                    else:
                        break  

        
    for spot in sorted_hex_spots_player2:
#        print('spot is:', spot)
        
        for d in range(6):
            spot_copy=copy.copy(spot)
#            print('direction is ',d)
            for i in range(5):
                cnt=0
                i=0
                while i<5:
                    next_spot=hex_neighbor(spot_copy,d)
#                    print('i',i)
#                    print('next_spot',next_spot)
                    if next_spot in sorted_hex_spots_player2:
                        spot_copy=next_spot
                        i+=1
                        cnt+=1
                        if cnt==4:
                            is_player2_win=True
                            break
                    else:
                        break 

    return is_player1_win, is_player2_win

def check_win_enclose(is_player1, is_player1_win, is_player2_win, steps, coordinates, coordmap):
    spots_player1=steps[::2]
    spots_player2=steps[1::2]
                
    hex_spots_player1=[]
    hex_spots_player2=[]
    
    
    for i in spots_player1:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player1.append(hex_value)
        
    for i in spots_player2:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player2.append(hex_value)    
        
    if is_player1==True:
        self_color=1
        opposite_color=2
        check_spot_hex=hex_spots_player1[-1]#last step of player 1
    else:
        self_color=2
        opposite_color=1
        check_spot_hex=hex_spots_player2[-1]#last step of player 2
        
        
    for d in range(6):
        next_spot=hex_neighbor(check_spot_hex,d)
        print('check_spot_hex is',check_spot_hex,' and next_spot is',next_spot)
        next_spot_value=coordmap[coordinates.index([next_spot.q,next_spot.r,next_spot.s])]
        if next_spot_value==opposite_color:#Color of Opponent
            visited=[]
            surrounded=[]
            enclosed=check_be_enclosed_flood_fill(is_player1, next_spot, steps, coordinates, coordmap,visited)
            print('enclosed ',enclosed,'for d',d)
            if enclosed==True and is_player1==True:
                is_player1_win=True
                break
            if enclosed==True and is_player1==False:
                is_player2_win=True
                break
    
                    
    return is_player1_win, is_player2_win
            
def check_be_enclosed_flood_fill(is_player1, next_spot,steps, coordinates, coordmap, visited):
    print('visited is',visited)
    enclosed=False

    visited.append(next_spot)
    next_spots=[]
    next_spots_value=[]
    
    if is_player1==True:
        self_color=1
        opposite_color=2
    else:
        self_color=2
        opposite_color=1

    for d in range(6):
        if d not in visited:    
            next_spots.append(hex_neighbor(next_spot,d))
            next_spots_value.append(coordmap[coordinates.index([hex_neighbor(next_spot,d).q,hex_neighbor(next_spot,d).r,hex_neighbor(next_spot,d).s])])
            
    if any(value == 0 for value in next_spots_value):
        return enclosed
    if all(value == self_color for value in next_spots_value):
        enclosed=True
        return enclosed
    if any(value == opposite_color for value in next_spots_value):
        next_spot= next_spots[next_spots_value.index(2)]
        enclosed=check_be_enclosed_flood_fill(is_player1, next_spot,steps, coordinates, coordmap, visited)
        return enclosed
    
    


        
    return enclosed  
    
    
    
def find_best_move_minimax(is_player1,steps,coordinates,coordmap,valid_moves):#Check parameters
    spots_player1=steps[::2]
    spots_player2=steps[1::2]
    best_move=Hex(0, 0, 0)           
    hex_spots_player1=[]
    hex_spots_player2=[]
    
    for i in spots_player1:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player1.append(hex_value)
        
    for i in spots_player2:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player2.append(hex_value)
        
        depth=1
        score, best_move=minimax(is_player1, coordinates,coordmap, depth, valid_moves, hex_spots_player1, hex_spots_player2)
    
    return best_move
      

def find_best_move_alphabeta(is_player1,steps,coordinates,coordmap,valid_moves):#Check parameters
    spots_player1=steps[::2]
    spots_player2=steps[1::2]
    best_move=Hex(0, 0, 0)           
    hex_spots_player1=[]
    hex_spots_player2=[]
    
    for i in spots_player1:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player1.append(hex_value)
        
    for i in spots_player2:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player2.append(hex_value)
        
        alpha=-2000
        beta=2000
        depth=1
        score, best_move=alphabeta(is_player1, coordinates,coordmap, depth, valid_moves, hex_spots_player1, hex_spots_player2, alpha, beta)
    return best_move

def minimax(is_player1, coordinates, coordmap, depth, valid_moves, hex_spots_player1, hex_spots_player2):
    if depth==0:
        board_score=calculate_board_score(is_player1,steps, coordinates, coordmap)
        return board_score, None
    
    scores=[]
    moves=[]
    hex_spots_player1_copy=copy.copy(hex_spots_player1)
    hex_spots_player2_copy=copy.copy(hex_spots_player2)
    
    for move in valid_moves:
        if is_player1==True:
            hex_spots_player1_copy.append(move)
        else:
            hex_spots_player2_copy.append(move)
        
        is_player1=not is_player1
        score, something = minimax(is_player1, coordinates, coordmap, depth-1, valid_moves, hex_spots_player1_copy, hex_spots_player2_copy)
        scores.append(score)
        moves.append(move)
        
        if len(scores) == 0:
            return

    if is_player1 == True:
        max_score_index = scores.index(max(scores))
        best_move = moves[max_score_index]
        return scores[max_score_index], best_move

    else:
        min_score_index = scores.index(min(scores))
        worst_opponent_move = moves[min_score_index]
        return scores[min_score_index], worst_opponent_move
    
def alphabeta(is_player1, coordinates, coordmap, depth, valid_moves, hex_spots_player1, hex_spots_player2, alpha, beta):
    
    if depth==0:
        board_score=calculate_board_score(is_player1,steps, coordinates, coordmap)
        return board_score, None
    
    scores=[]
    moves=[]
    
    if is_player1==True:
        for move in valid_moves:
            hex_spots_player1_copy=copy.copy(hex_spots_player1)
            hex_spots_player2_copy=copy.copy(hex_spots_player2)
            
            hex_spots_player1_copy.append(move)

            is_player1=not is_player1 #Calculate score according to next player
            
            score, something = alphabeta(is_player1, coordinates, coordmap, depth-1, valid_moves, hex_spots_player1_copy, hex_spots_player2_copy, alpha, beta)
            
            scores.append(score)
            moves.append(move)
            alpha=max(score, alpha)
            if beta <= alpha:
                break
        if len(scores)==0:
            return
        
        max_score_index = scores.index(max(scores))
        best_move = moves[max_score_index]
        return scores[max_score_index], best_move
    
    else:
        for move in valid_moves:
            hex_spots_player1_copy=copy.copy(hex_spots_player1)
            hex_spots_player2_copy=copy.copy(hex_spots_player2)

            hex_spots_player2_copy.append(move)
            
            is_player1=not is_player1 #Calculate score according to next player
            score, something = alphabeta(is_player1, coordinates, coordmap, depth-1, valid_moves, hex_spots_player1_copy, hex_spots_player2_copy, alpha, beta)
            
            scores.append(score)
            moves.append(move)
            beta=max(score, alpha)
            if beta <= alpha:
                break
        if len(scores)==0:
            return
        min_score_index = scores.index(min(scores))
        worst_opponent_move = moves[min_score_index]
        return scores[min_score_index], worst_opponent_move
   

def calculate_board_score(is_player1,steps, coordinates, coordmap):
    
    spots_player1=steps[::2]
    spots_player2=steps[1::2]
                
    hex_spots_player1=[]
    hex_spots_player2=[]
    
    score_player1=[0,0,0,0] #2_in_a_row,3_in_a_row,4_in_a_row,
    score_player2=[0,0,0,0]
    weight=[1,10,100,float("inf")]
    
    for i in spots_player1:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player1.append(hex_value)
        
    for i in spots_player2:
        hex_value=Hex(coordinates[i][0], coordinates[i][1], coordinates[i][2])
        hex_spots_player2.append(hex_value)    

    sorted_hex_spots_player1=sorted(hex_spots_player1) #Sorted in minq, minr, mins
    sorted_hex_spots_player2=sorted(hex_spots_player2)
    

    for i in range(2,6):
#    for spot in sorted_hex_spots_player1:
        for spot in sorted_hex_spots_player1:
#        for i in range(2,5):
            for d in range(6):
                spot_copy=copy.copy(spot)
                cnt=1
#                visited=0
#                a=0#length
#                print('check i,spot_copy,d,cnt: ',i,spot_copy,d,cnt)
                while cnt<=i:
                    next_spot=hex_neighbor(spot_copy,d)
                    updated_sorted_hex_spots_player1=copy.copy(sorted_hex_spots_player1)
#                    print(updated_sorted_hex_spots_player1,)
                    updated_sorted_hex_spots_player1.remove(spot_copy)
#                    print('after pop: ',updated_sorted_hex_spots_player1)#Checked
#                    print('spot_copy',spot_copy,'next_spot',next_spot)
#                    print('updated_sorted_hex_spots_player1',updated_sorted_hex_spots_player1)
                    if next_spot in sorted_hex_spots_player1:#updated_sorted_hex_spots_player1:
#                        if d<3:
#                            visited_d=(d+3)
#                        else:
#                            visited_d=(d+3-6)
                        spot_copy=next_spot
#                        a+=1
                        cnt+=1
#                        print('NOW in list, so check i,spot_copy,d,cnt:',i,spot_copy,d,cnt)
                        following_spot_value=[0]
                        following_spot_test=next_spot
                        for test in range(5-i):
                            test+=1
                            following_spot_test=hex_neighbor(following_spot_test,d)
                            following_spot_value.append(coordmap[coordinates.index([following_spot_test.q,following_spot_test.r,following_spot_test.s])])

#                            print('following_spot_test is:',following_spot_test)
#                            print('following_spot is:',following_spot)
                        if cnt==(i) and all(v == 0 for v in following_spot_value)==True:
                            score_player1[i-2]+=1
#                            print('break with:',i,d,spot_copy,cnt)
                            break #Break the while loop
                    else:
#                        print(next_spot,'is not in updated_list, break the while loop cnt<=i')
                        break #Break the while loop for cnt<=i
    
    for i in range(2,6):
        for spot in sorted_hex_spots_player2:
            for d in range(6):
                spot_copy=copy.copy(spot)
                cnt=1
                while cnt<=i:
                    next_spot=hex_neighbor(spot_copy,d)
                    updated_sorted_hex_spots_player2=copy.copy(sorted_hex_spots_player2)
                    updated_sorted_hex_spots_player2.remove(spot_copy)
                    if next_spot in sorted_hex_spots_player2:
                        spot_copy=next_spot
                        cnt+=1
                        following_spot_value=[0]
                        following_spot_test=next_spot
                        for test in range(5-i):
                            test+=1
                            following_spot_test=hex_neighbor(following_spot_test,d)
                            following_spot_value.append(coordmap[coordinates.index([following_spot_test.q,following_spot_test.r,following_spot_test.s])])

                        if cnt==(i) and all(v == 0 for v in following_spot_value)==True:
                            score_player2[i-2]+=1
                            break #Break the while loop
                    else:
                        break #Break the while loop for cnt<=i
                        
    
    Total_score_1=weight[0]*score_player1[0]+weight[1]*score_player1[1]+weight[2]*score_player1[2]
    Total_score_2=weight[0]*score_player2[0]+weight[1]*score_player2[1]+weight[2]*score_player2[2]
    print('score_player1 is',score_player1)
    print('score_player2 is',score_player2)    
    print('Total_score_1 is',Total_score_1)
    print('Total_score_2 is',Total_score_2)
    if is_player1==True:
        score=Total_score_1
    else:
        score=Total_score_2
        
    return score

class Map():
    def __init__(self, steps):
        self.steps=steps
#        self.width = width
#        self.height = height
    
#    def reverseTurn(self, turn):
#        if turn == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
#            return MAP_ENTRY_TYPE.MAP_PLAYER_TWO
#        else:
#            return MAP_ENTRY_TYPE.MAP_PLAYER_ONE
#        
    def HexToPixel(self, coord):
        map_x, map_y = hex_to_pixel(pointy, coord)       
        return (map_x, map_y)
        
    def PixelToHex(self,x, y):
        coord = hex_round(pixel_to_hex(pointy, Point(x, y)))
        return (coord) #return hex coordinate
    
#    def isInMap(self, map_x, map_y):
#        if (map_x <= 0 or map_x >= MAP_WIDTH or 
#            map_y <= 0 or map_y >= MAP_HEIGHT):
#            return False
#        return True
#
#    def isEmpty(self, coord):
#        return (self.coordmap[coordinates.index(coord)] == 0)
#    
#    def click(self, x, y, type):
#        coord=hex_round(pixel_to_hex(pointy, Point(y, x)))  #Check x and y positions
#        self.coordmap[coordinates.index(coord)] = type.value   #Note here! Inverse!
#        self.steps.append((coord))

    def drawChess(self, screen, steps, coordmap):
        player_one = (0, 0, 0)
        player_two = (255, 255, 255)
        player_color = [player_one, player_two]
        
#        font = pygame.font.SysFont(None, 2)
        for i in range(len(steps)):
            coord_hex = Hex(coordinates[steps[i]][0], coordinates[steps[i]][1], coordinates[steps[i]][2])
            coord = coordinates[steps[i]] #Get the Hex coordinates of the step
            corner=polygon_corners(pointy, coord_hex)
            turn = coordmap[coordinates.index(coord)]
#            print('turn is:',turn)
#            print('coordmap is:',coordmap)
            pygame.gfxdraw.filled_polygon(screen, corner,player_color[turn-1])
            
#            if turn == 1:
#                op_turn = 2
#            else:
#                op_turn = 1
#
#            #Don't understand msg_image
#            msg_image = font.render(str(i), True, player_color[op_turn-1], player_color[turn-1])
#            msg_image_rect = msg_image.get_rect()
#            msg_image_rect.center = pos
#            screen.blit(msg_image, msg_image_rect)
            
#        #Draw rectangle for last step
#        if len(self.steps) > 0:
#            last_pos = self.steps[-1]
#            map_x, map_y, width, height = self.getMapUnitRect(last_pos[0], last_pos[1])
#            purple_color = (255, 0, 255)
#            point_list = [(map_x, map_y), (map_x + width, map_y), 
#                    (map_x + width, map_y + height), (map_x, map_y + height)]
#            pygame.draw.lines(screen, purple_color, True, point_list, 1)


    def drawBestMove(self, screen, best_move, coordmap):
        best_move_color=(34,139,34)
        coord_hex = best_move
        corner=polygon_corners(pointy, coord_hex)
        pygame.gfxdraw.filled_polygon(screen, corner, best_move_color)

    def drawValidMoves(self, screen, valid_moves, coordmap):
        valid_moves_color=(255,0,0)
        for i in range(len(valid_moves)):
            coord_hex = valid_moves[i]
            corner=polygon_corners(pointy, coord_hex)
            pygame.gfxdraw.polygon(screen, corner,valid_moves_color)
            
        
    def drawBackground(self, screen, corners):
        color = (255,255,255)
        for corner in corners:
            pygame.gfxdraw.polygon(screen, corner, color)
            





#Game Map Design

CHESS_LEN=3

REC_SIZE = 3
CHESS_RADIUS = 1


MAP_WIDTH = 550
MAP_HEIGHT = 650 # For 9*9

INFO_WIDTH = 200
BUTTON_WIDTH = 140
BUTTON_HEIGHT = 50

SCREEN_WIDTH = MAP_WIDTH + INFO_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT

size = 9 
coordinates=[] # For board hex coordrdinates
corners=[] #For corners of hex coordinates
pixels=[] #For board pixel

pointy = Layout(layout_pointy, Point(15.0, 15.0), Point(250.0, 330.0)) #Layout, size, origin(fits to 9*9)

for x in range(-size,size+1):
    for y in range(-size,size+1):
        for z in range(-size,size+1):
            if x == -size and y == -size and z == -size:
                continue
            if x+y+z == 0:
                coordinates.append([x,y,z])
                
for a in coordinates:
    coord=Hex(a[0],a[1], a[2]) #Change to type:Hex
    x,y=hex_to_pixel(pointy,coord)
    corner=polygon_corners(pointy, coord)
    corners.append(corner) #Collect corners of all Hexes
    pixels.append([x,y]) #Screen pixels of Hexes
    

#class MAP_ENTRY_TYPE(IntEnum):
#    MAP_EMPTY = 0,
#    MAP_PLAYER_ONE = 1,
#    MAP_PLAYER_TWO = 2,
#    MAP_NONE = 3, # out of map range
    
EMPTY = 0 #Define turn
BLACK = 1
WHITE = 2


steps = [] 

def main():
    board=Map(steps)
    is_player1 = True
    pygame.init()
    pygame.display.set_caption('Andantino')

    screen = pygame.display.set_mode((MAP_WIDTH,MAP_HEIGHT))
    
    screen.fill([130,100,30]) 
    board.drawBackground(screen, corners) 
    pygame.display.flip() 
    
    running = True
    coordmap=[0 for x in range(271)] #Coordinate of steps from pixel. 271 is for 9*9 board.  0 means no player in this spot.

    idx=999 #Initialize idx
    is_player1_win = False
    is_player2_win = False
    while running:
        for event in pygame.event.get():
          
          if event.type == pygame.QUIT:
            running = False
     
          elif event.type == pygame.KEYUP:
            pass
          elif event.type == pygame.MOUSEBUTTONDOWN and \
          event.button == 1:# left button
              x, y = event.pos 
#              print('here is clicked x,y',x,y)
              coord = board.PixelToHex(x, y) #After check, the correct order is x,y
#              print('corresponding hex coord is', coord, type(coord))
              if list(coord) not in coordinates: #Checked
                  print('not in board')
                  continue
              if coordmap[coordinates.index(list(coord))]!=0: #Checked
                  print('spot occupied')
                  continue
              idx=coordinates.index(list(coord))

              steps.append(idx)
              
              if is_player1 == True:
                  coordmap[idx]=1 #Player one in this spot
              else:
                  coordmap[idx]=2 #Player two in this spot


              #Check win
              is_player1_win, is_player2_win = check_win_fiveinarow(is_player1_win, is_player2_win, steps, coordinates, coordmap)
              if len(steps)>10:
                  is_player1_win, is_player2_win = check_win_enclose(is_player1, is_player1_win, is_player2_win, steps, coordinates, coordmap)
              
#              print('is_player1_win is:',is_player1_win)
#              print('is_player2_win is:',is_player2_win)                

              #Find best move
              
              board.drawChess(screen, steps, coordmap)
              
              is_player1 = not is_player1 #Change player
              
              if len(steps)>1:
                  valid_moves=find_legal_move(steps, coordinates, coordmap) 
#                  print('valid_move is:',valid_moves)
                  best_move=find_best_move_alphabeta(is_player1,steps,coordinates,coordmap,valid_moves) #For next player
#                  print('best_move',best_move)
                  best_move=find_best_move_minimax(is_player1,steps,coordinates,coordmap,valid_moves)
                  board.drawValidMoves(screen, valid_moves, coordmap)
                  board.drawBestMove(screen, best_move, coordmap)

              


              if is_player1_win is True:
                  font = pygame.font.Font(pygame.font.get_default_font(), 36)
                  text_surface = font.render('Player1 (black) Wins!', True, (34,139,34))
                  screen.blit(text_surface, dest=(0,0))
                  

              
              if is_player2_win is True:
                  font = pygame.font.Font(pygame.font.get_default_font(), 36)
                  text_surface = font.render('Player2 (black) Wins!', True, (34,139,34))
                  screen.blit(text_surface, dest=(0,0))


              pygame.display.flip()
              
                                   
    pygame.quit()

if __name__ == '__main__':
  main()