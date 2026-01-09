import pygame
import random
import time
from queue import Queue, PriorityQueue

# CONFIGURATION
CELL_SIZE = 25
ROWS = 25
COLS = 25
PANEL_WIDTH = 500
WINDOW_WIDTH = CELL_SIZE * COLS + PANEL_WIDTH
WINDOW_HEIGHT = max(CELL_SIZE * ROWS, 650) 
FPS = 100

# ---  COLOR PALETTE
BG_MAIN = (30, 33, 40)       
BG_PANEL = (40, 44, 52)      
BG_TILE = (46, 52, 64)       

WALL_COLOR = (216, 222, 233) 
WALL_SHADOW = (20, 20, 25)   

COLOR_MUD = (163, 114, 88)   
COLOR_WATER = (94, 129, 172) 
COLOR_TRAP = (191, 97, 106)  

COLOR_START = (163, 190, 140) 
COLOR_GOAL = (235, 203, 139)  
COLOR_PLAYER = (136, 192, 208) 
COLOR_PLAYER_BORDER = (230, 230, 230)

COLOR_BFS = (136, 192, 208)  
COLOR_DFS = (208, 135, 112)  
COLOR_UCS = (235, 203, 139)  
COLOR_BEST = (143, 188, 187) 
COLOR_ASTAR = (180, 142, 173) 

TEXT_WHITE = (236, 239, 244)
TEXT_GREY = (180, 180, 190)
BTN_DEFAULT = (67, 76, 94)
BTN_HOVER = (76, 86, 106)

TERRAIN_COST = {0: 1, 2: 3, 3: 5, 4: 10}


# MAZE GENERATION

def generate_maze(rows, cols):
    maze = [[1 for i in range(cols)] for i in range(rows)]
    def get_neighbors(r, c):
        neighbors = []
        directions = [(-2,0),(2,0),(0,-2),(0,2)]
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if 1<=nr<rows-1 and 1<=nc<cols-1:
                neighbors.append((nr,nc))
        return neighbors

    maze[1][1]=0
    walls = [( (1,1), n ) for n in get_neighbors(1,1)]
    while walls:
        wall = random.choice(walls)
        walls.remove(wall)
        (r1,c1),(r2,c2)=wall
        if maze[r2][c2]==1:
            maze[r2][c2]=0
            maze[(r1+r2)//2][(c1+c2)//2]=0
            for n in get_neighbors(r2,c2):
                if maze[n[0]][n[1]]==1:
                    walls.append(((r2,c2), n))
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            if maze[r][c]==1 and random.random()<0.05:
                if ((maze[r-1][c]==0 and maze[r+1][c]==0) or (maze[r][c-1]==0 and maze[r][c+1]==0)):
                    maze[r][c]=0
    maze[1][1]=0
    maze[rows-2][cols-2]=0
    maze = add_terrain_weights(maze)
    return maze

def add_terrain_weights(maze):
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c]==0:
                rand = random.random()
                if rand<0.05: 
                    maze[r][c]=2 
                elif rand<0.08: 
                    maze[r][c]=3 
                elif rand<0.10: 
                    maze[r][c]=4 
    return maze


# HELPERS & SEARCH ALGORITHMS

def get_neighbors(node, maze):
    neighbors = []
    x, y = node
    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0<=nx<ROWS and 0<=ny<COLS and maze[nx][ny]!=1:
            neighbors.append((nx,ny))
    return neighbors

def reconstruct_path(came_from,start,end):
    path=[]
    node=end
    if node not in came_from: return []
    while node!=start:
        path.append(node)
        node=came_from[node]
    path.append(start)
    path.reverse()
    return path

def bfs(maze,start,goal):
    start_time=time.time()
    frontier=Queue()
    frontier.put(start)
    came_from={start:None}
    visited_nodes=0
    while not frontier.empty():
        current=frontier.get()
        visited_nodes+=1
        if current==goal:
            path=reconstruct_path(came_from,start,goal)
            return path,visited_nodes,time.time()-start_time,len(path),len(path)
        for n in get_neighbors(current,maze):
            if n not in came_from:
                frontier.put(n)
                came_from[n]=current
    return [],visited_nodes,time.time()-start_time,0,0

def dfs(maze,start,goal):
    start_time=time.time()
    frontier=[start]
    came_from={start:None}
    visited_nodes=0
    while frontier:
        current=frontier.pop()
        visited_nodes+=1
        if current==goal:
            path=reconstruct_path(came_from,start,goal)
            return path,visited_nodes,time.time()-start_time,len(path),len(path)
        for n in get_neighbors(current,maze):
            if n not in came_from:
                frontier.append(n)
                came_from[n]=current
    return [],visited_nodes,time.time()-start_time,0,0

def ucs(maze,start,goal):
    start_time=time.time()
    frontier=PriorityQueue()
    frontier.put((0,start))
    came_from={start:None}
    cost_so_far={start:0}
    visited_nodes=0
    while not frontier.empty():
        current_cost,current=frontier.get()
        visited_nodes+=1
        if current==goal:
            path=reconstruct_path(came_from,start,goal)
            return path,visited_nodes,time.time()-start_time,len(path),cost_so_far[goal]
        for n in get_neighbors(current,maze):
            terrain=maze[n[0]][n[1]]
            new_cost=cost_so_far[current]+TERRAIN_COST.get(terrain,1)
            if n not in cost_so_far or new_cost<cost_so_far[n]:
                cost_so_far[n]=new_cost
                frontier.put((new_cost,n))
                came_from[n]=current
    return [],visited_nodes,time.time()-start_time,0,0

def astar(maze,start,goal):
    def h(a,b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])
    start_time=time.time()
    frontier=PriorityQueue()
    frontier.put((0 + h(start,goal), start))
    came_from={start:None}
    cost_so_far={start:0}
    visited_nodes=0
    while not frontier.empty():
        _,current=frontier.get()
        visited_nodes+=1
        if current==goal:
            path=reconstruct_path(came_from,start,goal)
            return path,visited_nodes,time.time()-start_time,len(path),cost_so_far[goal]
        for n in get_neighbors(current,maze):
            terrain=maze[n[0]][n[1]]
            new_cost=cost_so_far[current]+TERRAIN_COST.get(terrain,1)
            if n not in cost_so_far or new_cost<cost_so_far[n]:
                cost_so_far[n]=new_cost
                frontier.put((new_cost+h(n,goal),n))
                came_from[n]=current
    return [],visited_nodes,time.time()-start_time,0,0

def best_first(maze,start,goal):
    def h(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    start_time=time.time()
    frontier=PriorityQueue()
    frontier.put((h(start,goal),start))
    came_from={start:None}
    visited_nodes=0
    visited = {start}
    while not frontier.empty():
        _,current=frontier.get()
        visited_nodes+=1
        if current==goal:
            path=reconstruct_path(came_from,start,goal)
            return path,visited_nodes,time.time()-start_time,len(path),len(path)
        for n in get_neighbors(current,maze):
            if n not in visited:
                visited.add(n)
                came_from[n]=current
                frontier.put((h(n,goal),n))
    return [],visited_nodes,time.time()-start_time,0,0

def move_player(pos,key,maze):
    x,y=pos
    if key==pygame.K_UP and x>0 and maze[x-1][y]!=1: 
        x-=1
    elif key==pygame.K_DOWN and x<ROWS-1 and maze[x+1][y]!=1:
        x+=1
    elif key==pygame.K_LEFT and y>0 and maze[x][y-1]!=1:
        y-=1
    elif key==pygame.K_RIGHT and y<COLS-1 and maze[x][y+1]!=1:
        y+=1
    return (x,y)


# UI & BUTTON CLASS

class Button:
    def __init__(self,x,y,w,h,text,color=BTN_DEFAULT,text_color=TEXT_WHITE,action=None):
        self.rect=pygame.Rect(x,y,w,h)
        self.text=text
        self.base_color=color
        self.text_color=text_color
        self.action=action
        self.active=False
        
    def draw(self,screen,font,mouse_pos):
        is_hovered = self.rect.collidepoint(mouse_pos)
        
        if self.active:
            if "bfs" in self.action: 
                color = COLOR_BFS
            elif "dfs" in self.action: 
                color = COLOR_DFS
            elif "ucs" in self.action: 
                color = COLOR_UCS
            elif "best" in self.action: 
                color = COLOR_BEST
            elif "astar" in self.action: 
                color = COLOR_ASTAR
            elif "stats" in self.action: 
                color = COLOR_START
            else: color = (100, 100, 120)
        elif is_hovered:
            color = BTN_HOVER
        else:
            color = self.base_color

        pygame.draw.rect(screen, (20,20,25), (self.rect.x, self.rect.y+3, self.rect.w, self.rect.h), border_radius=8)
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        
        text_surf = font.render(self.text, True, (30,30,35) if self.active else self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def is_clicked(self,pos):
        return self.rect.collidepoint(pos)


# DRAW MAZE 

def draw_maze(screen, maze, active_paths, player_pos, font):
    maze_width = COLS * CELL_SIZE
    maze_height = ROWS * CELL_SIZE
    
    pygame.draw.rect(screen, BG_MAIN, (0, 0, maze_width, maze_height))
    
    tile_padding = 2
    for i in range(ROWS):
        for j in range(COLS):
            val = maze[i][j]
            x, y = j * CELL_SIZE, i * CELL_SIZE
            tile_color = BG_TILE
            if val == 2: tile_color = COLOR_MUD
            elif val == 3: tile_color = COLOR_WATER
            elif val == 4: tile_color = COLOR_TRAP
            
            if val != 1:
                pygame.draw.rect(screen, tile_color, 
                                (x + tile_padding, y + tile_padding, 
                                 CELL_SIZE - tile_padding*2, CELL_SIZE - tile_padding*2), 
                                border_radius=4)

    wall_thickness = int(CELL_SIZE * 0.25) 
    offset = (CELL_SIZE - wall_thickness) // 2

    def draw_wall_segment(r, c, w, h, is_shadow=False):
        x = c * CELL_SIZE + offset
        y = r * CELL_SIZE + offset
        if is_shadow:
            pygame.draw.rect(screen, WALL_SHADOW, (x+2, y+2, w, h), border_radius=2)
        else:
            pygame.draw.rect(screen, WALL_COLOR, (x, y, w, h), border_radius=2)

    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == 1:
                draw_wall_segment(r, c, wall_thickness, wall_thickness, True)
                if c + 1 < COLS and maze[r][c+1] == 1:
                    draw_wall_segment(r, c, CELL_SIZE - offset + 2, wall_thickness, True)
                if r + 1 < ROWS and maze[r+1][c] == 1:
                    draw_wall_segment(r, c, wall_thickness, CELL_SIZE - offset + 2, True)
# main wall
    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == 1:
                if c + 1 < COLS and maze[r][c+1] == 1:
                    pygame.draw.rect(screen, WALL_COLOR, 
                        (c*CELL_SIZE + offset, r*CELL_SIZE + offset, CELL_SIZE, wall_thickness), border_radius=2)
                if r + 1 < ROWS and maze[r+1][c] == 1:
                    pygame.draw.rect(screen, WALL_COLOR, 
                        (c*CELL_SIZE + offset, r*CELL_SIZE + offset, wall_thickness, CELL_SIZE), border_radius=2)
                pygame.draw.rect(screen, WALL_COLOR, 
                        (c*CELL_SIZE + offset, r*CELL_SIZE + offset, wall_thickness, wall_thickness), border_radius=2)

    start_rect = pygame.Rect(1*CELL_SIZE + 4, 1*CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
    pygame.draw.rect(screen, COLOR_START, start_rect, border_radius=6)
    
    goal_rect = pygame.Rect((COLS-2)*CELL_SIZE + 4, (ROWS-2)*CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
    pygame.draw.rect(screen, COLOR_GOAL, goal_rect, border_radius=6)
    
    for path, color in active_paths:
        if len(path) > 1:
            points = [(c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2) for r, c in path]
            if len(points) >= 2:
                pygame.draw.lines(screen, color, False, points, 4)
            
    px, py = player_pos
    cx, cy = py*CELL_SIZE + CELL_SIZE//2, px*CELL_SIZE + CELL_SIZE//2
    pygame.draw.circle(screen, (0,0,0), (cx+2, cy+2), CELL_SIZE//3)
    pygame.draw.circle(screen, COLOR_PLAYER_BORDER, (cx, cy), CELL_SIZE//3)
    pygame.draw.circle(screen, COLOR_PLAYER, (cx, cy), CELL_SIZE//3 - 2)

# GRAPHICS: UI & LEGEND

def draw_terrain_legend(screen, font):
    legend_items = [
        ("Start", COLOR_START),
        ("End", COLOR_GOAL),
        ("Player", COLOR_PLAYER), 
        ("Wall", WALL_COLOR),
        ("Path", BG_TILE), 
        ("Mud", COLOR_MUD),
        ("Water", COLOR_WATER), 
        ("Trap", COLOR_TRAP)
    ]
    
    # near the bottom of the side panel
    start_y = WINDOW_HEIGHT - 150
    x = COLS * CELL_SIZE + 30 #right
    box_size = 18
    col_spacing = 140   
    row_spacing = 30
    
    # Header + separator line
    pygame.draw.line(screen, BTN_DEFAULT, (x, start_y - 15), (x + 220, start_y - 15), 1)
    lbl = font.render("TERRAIN & COSTS", True, TEXT_GREY)
    screen.blit(lbl, (x, start_y - 35))

    for i, (name, color) in enumerate(legend_items):
        col = i // 4    # 2 columns
        row = i % 4
        
        current_x = x + col * col_spacing
        current_y = start_y + row * row_spacing

        # Color box
        pygame.draw.rect(screen, color, (current_x, current_y, box_size, box_size), border_radius=4)
        if name == "Path":
            # outline to make path clearer on dark background
            pygame.draw.rect(screen, TEXT_GREY, (current_x, current_y, box_size, box_size), 1, border_radius=4)

        #  label
        name_surf = font.render(name, True, TEXT_WHITE)
        screen.blit(name_surf, (current_x + box_size + 8, current_y))

        
        cost_text = ""
        if name == "Path":
            cost_text = f"Cost: {TERRAIN_COST.get(0, 1)}"
        elif name == "Mud":
            cost_text = f"Cost: {TERRAIN_COST.get(2, 1)}"
        elif name == "Water":
            cost_text = f"Cost: {TERRAIN_COST.get(3, 1)}"
        elif name == "Trap":
            cost_text = f"Cost: {TERRAIN_COST.get(4, 1)}"

        if cost_text:
            cost_surf = font.render(cost_text, True, TEXT_GREY)
            # Put cost on the same row, slightly to the right
            screen.blit(cost_surf, (current_x + box_size + 60, current_y + 0))



def draw_ui(screen, font, title_font, buttons, game_state, player_time):
    panel_rect = pygame.Rect(COLS*CELL_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, BG_PANEL, panel_rect)
    
    pygame.draw.line(screen, (20,20,25), (COLS*CELL_SIZE, 0), (COLS*CELL_SIZE, WINDOW_HEIGHT), 2)

    title = title_font.render("MAZE SOLVER AI", True, TEXT_WHITE)
    screen.blit(title, (COLS*CELL_SIZE + 30, 30))
    
    sub = font.render("Pathfinding Visualizer", True, COLOR_PLAYER)
    screen.blit(sub, (COLS*CELL_SIZE + 30, 60))

    y_offset = 110
    
    if game_state == "PLAYING":
        pygame.draw.rect(screen, BG_MAIN, (COLS*CELL_SIZE + 20, y_offset, PANEL_WIDTH - 40, 100), border_radius=10)
        
        lines = ["Navigate with Arrows", "Reach the Yellow Goal", "Avoid Red Traps"]
        for i, line in enumerate(lines):
            text = font.render(f"• {line}", True, TEXT_GREY)
            screen.blit(text, (COLS*CELL_SIZE + 40, y_offset + 15 + i*25))
        
        y_offset += 120
        
        time_label = font.render("CURRENT TIME", True, TEXT_GREY)
        screen.blit(time_label, (COLS*CELL_SIZE + 30, y_offset))
        
        time_val = title_font.render(f"{player_time:.1f}s", True, COLOR_PLAYER)
        screen.blit(time_val, (COLS*CELL_SIZE + 30, y_offset + 25))
        
        draw_terrain_legend(screen, font)

    elif game_state == "FINISHED":
        # success Box
        box_h = 75
        pygame.draw.rect(screen, (40, 60, 50), (COLS*CELL_SIZE + 20, y_offset, PANEL_WIDTH - 40, box_h), border_radius=10)
        pygame.draw.rect(screen, COLOR_START, (COLS*CELL_SIZE + 20, y_offset, PANEL_WIDTH - 40, box_h), 2, border_radius=10)
        
        msg = title_font.render("LEVEL COMPLETE", True, COLOR_START)
        screen.blit(msg, (COLS*CELL_SIZE + 40, y_offset + 12))
        
        time_text = font.render(f"Final Time: {player_time:.2f}s", True, TEXT_WHITE)
        screen.blit(time_text, (COLS*CELL_SIZE + 40, y_offset + 42))
        
        draw_terrain_legend(screen, font)
        
        # buttons are positioned between the Success Box and the Legend.
        mouse_pos = pygame.mouse.get_pos()
        for btn in buttons:
            btn.draw(screen, font, mouse_pos)


# STATS OVERLAY

def draw_stats_overlay(screen, font, title_font, player_data, bfs_data, dfs_data, ucs_data, best_data, astar_data):
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((15, 18, 25, 240)) 
    
    chart_w, chart_h = 700, 500
    chart_x = (WINDOW_WIDTH - chart_w) // 2
    chart_y = (WINDOW_HEIGHT - chart_h) // 2
    
    pygame.draw.rect(overlay, BG_PANEL, (chart_x, chart_y, chart_w, chart_h), border_radius=15)
    pygame.draw.rect(overlay, WALL_COLOR, (chart_x, chart_y, chart_w, chart_h), 1, border_radius=15)
    
    title = title_font.render("Algorithm Performance Comparison", True, TEXT_WHITE)
    overlay.blit(title, (chart_x + 30, chart_y + 25))

    p_time, p_steps = player_data
    def calc_score(t, l, c, n):
        cost_metric = c if c > 0 and (t < 0.1 or n < 1000) else l
        return max(0, int(1000 - t*500 - cost_metric*2 - n*0.1))

    bfs_score = calc_score(bfs_data[2], bfs_data[3], bfs_data[4], bfs_data[1])
    dfs_score = calc_score(dfs_data[2], dfs_data[3], dfs_data[4], dfs_data[1])
    ucs_score = calc_score(ucs_data[2], ucs_data[3], ucs_data[4], ucs_data[1])
    best_score = calc_score(best_data[2], best_data[3], best_data[4], best_data[1])
    ast_score = calc_score(astar_data[2], astar_data[3], astar_data[4], astar_data[1])
    p_score = max(0, int(1000 - p_time*10 - p_steps*2))

    algos = [
        ("Player", TEXT_WHITE), ("BFS", COLOR_BFS), ("DFS", COLOR_DFS), 
        ("UCS", COLOR_UCS), ("A*", COLOR_ASTAR)
    ]
    metric_data = [
        ("Time (s)", [p_time, bfs_data[2], dfs_data[2], ucs_data[2], astar_data[2]]),
        ("Path Len", [p_steps, bfs_data[3], dfs_data[3], ucs_data[3], astar_data[3]]),
        ("Nodes", [0, bfs_data[1], dfs_data[1], ucs_data[1], astar_data[1]]),
        ("Cost", [p_steps, bfs_data[4], dfs_data[4], ucs_data[4], astar_data[4]]),
        ("Score", [p_score, bfs_score, dfs_score, ucs_score, ast_score]),
    ]

    inner_x = chart_x + 100
    inner_y = chart_y + 80
    inner_w = chart_w - 140
    inner_h = chart_h - 140

    num_metrics = len(metric_data)
    panel_h = inner_h / num_metrics
    label_font = pygame.font.SysFont("Arial", 14)

    for m_idx, (metric_name, values) in enumerate(metric_data):
        top = inner_y + m_idx * panel_h
        bottom = top + panel_h - 30
        height_available = bottom - (top + 20)
        base_y = bottom

        m_text = font.render(metric_name, True, TEXT_GREY)
        overlay.blit(m_text, (chart_x + 20, top + 5))

        max_val = max(values)
        if max_val <= 0: max_val = 1.0

        slot_w = (inner_w) / len(algos)
        bar_w = slot_w * 0.5

        for i, ((_, color), val) in enumerate(zip(algos, values)):
            cx = inner_x + slot_w * i + slot_w / 2
            bar_h = int((val / max_val) * height_available)
            bar_x = int(cx - bar_w / 2)
            bar_y = int(base_y - bar_h)

            bar_rect = pygame.Rect(bar_x, bar_y, int(bar_w), bar_h)
            pygame.draw.rect(overlay, color, bar_rect, border_radius=4)
            
            val_str = f"{val:.2f}" if isinstance(val, float) and val < 1.0 else str(int(round(val)))
            val_text = label_font.render(val_str, True, TEXT_WHITE)
            overlay.blit(val_text, (bar_x + (bar_w - val_text.get_width())/2, bar_y - val_text.get_height() - 2))

        pygame.draw.line(overlay, BTN_DEFAULT, (inner_x, bottom), (inner_x + inner_w, bottom), 1)

    legend_y = chart_y + chart_h - 40
    legend_x = chart_x + 30
    all_algos = [("Player", TEXT_WHITE), ("BFS", COLOR_BFS), ("DFS", COLOR_DFS), ("UCS", COLOR_UCS), ("A*", COLOR_ASTAR), ("BestFS", COLOR_BEST)]
    
    for name, color in all_algos:
        pygame.draw.circle(overlay, color, (legend_x, legend_y), 5)
        lbl = label_font.render(name, True, TEXT_GREY)
        overlay.blit(lbl, (legend_x + 10, legend_y - 8))
        legend_x += 80

    close_text = font.render("Click anywhere to close", True, COLOR_PLAYER)
    overlay.blit(close_text, (chart_x + chart_w - close_text.get_width() - 20, chart_y + 25))
    
    screen.blit(overlay, (0, 0))


# MAIN FUNCTION

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Professional Maze AI")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("Verdana", 14)
    title_font = pygame.font.SysFont("Verdana", 20, bold=True)

    maze = generate_maze(ROWS, COLS)
    start_node = (1, 1)
    goal_node = (ROWS-2, COLS-2)
    player_pos = start_node
    
    bfs_res = bfs(maze, start_node, goal_node)
    dfs_res = dfs(maze, start_node, goal_node)
    ucs_res = ucs(maze, start_node, goal_node)
    best_res = best_first(maze, start_node, goal_node)
    astar_res = astar(maze, start_node, goal_node)

    # --- BUTTON LAYOUT ---
    panel_x = COLS * CELL_SIZE
    col1_x = panel_x + 30
    col2_x = panel_x + 240
    
    start_y = 210 # positioned right after lvl compl box
    btn_w = 190
    btn_h = 40
    gap_y = 50
    
    # Column 1
    btn_bfs = Button(col1_x, start_y, btn_w, btn_h, "Toggle BFS", action="toggle_bfs")
    btn_dfs = Button(col1_x, start_y + gap_y, btn_w, btn_h, "Toggle DFS", action="toggle_dfs")
    btn_ucs = Button(col1_x, start_y + gap_y*2, btn_w, btn_h, "Toggle UCS", action="toggle_ucs")
    
    # Column 2
    btn_best = Button(col2_x, start_y, btn_w, btn_h, "Toggle BestFS", action="toggle_best")
    btn_astar = Button(col2_x, start_y + gap_y, btn_w, btn_h, "Toggle A*", action="toggle_astar")
    
    # Stats 
    btn_stats = Button(col1_x, start_y + gap_y*3 + 10, 400, 45, "View Analytics & Compare", color=COLOR_START, text_color=(30,30,35), action="toggle_stats")
    
    finished_buttons = [btn_bfs, btn_dfs, btn_ucs, btn_best, btn_astar, btn_stats]

    show_bfs = show_dfs = show_ucs = show_best = show_astar = False
    show_stats_overlay = False

    game_state = "PLAYING"
    start_time = time.time()
    final_time = 0
    player_steps = 0

    running = True
    while running:
        current_time = time.time()
        elapsed = current_time - start_time if game_state=="PLAYING" else final_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if game_state=="PLAYING" and event.type==pygame.KEYDOWN:
                new_pos = move_player(player_pos, event.key, maze)
                if new_pos != player_pos:
                    player_steps += 1
                    player_pos = new_pos
                    if player_pos == goal_node:
                        game_state="FINISHED"
                        final_time=elapsed
                        
            if event.type==pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if show_stats_overlay:
                    show_stats_overlay = False
                    btn_stats.active = False
                elif game_state=="FINISHED":
                    for btn in finished_buttons:
                        if btn.is_clicked(mouse_pos):
                            if btn.action=="toggle_bfs":
                                show_bfs = not show_bfs
                                btn.active = show_bfs
                            elif btn.action=="toggle_dfs":
                                show_dfs = not show_dfs
                                btn.active = show_dfs
                            elif btn.action=="toggle_ucs":
                                show_ucs = not show_ucs
                                btn.active = show_ucs
                            elif btn.action=="toggle_best":
                                show_best = not show_best
                                btn.active = show_best
                            elif btn.action=="toggle_astar":
                                show_astar = not show_astar
                                btn.active = show_astar
                            elif btn.action=="toggle_stats":
                                show_stats_overlay = not show_stats_overlay
                                btn.active = show_stats_overlay

        active_paths=[]
        if show_bfs: active_paths.append((bfs_res[0], COLOR_BFS))
        if show_dfs: active_paths.append((dfs_res[0], COLOR_DFS))
        if show_ucs: active_paths.append((ucs_res[0], COLOR_UCS))
        if show_best: active_paths.append((best_res[0], COLOR_BEST))
        if show_astar: active_paths.append((astar_res[0], COLOR_ASTAR))

        draw_maze(screen, maze, active_paths, player_pos, font)
        draw_ui(screen, font, title_font, finished_buttons if game_state=="FINISHED" else [], game_state, elapsed)
        
        if game_state=="FINISHED" and show_stats_overlay:
            player_data = (final_time, player_steps)
            draw_stats_overlay(screen, font, title_font, player_data, bfs_res, dfs_res, ucs_res, best_res, astar_res)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__=="__main__":
    main()