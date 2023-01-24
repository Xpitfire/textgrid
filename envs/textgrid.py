import argparse
import os
from colour import Color
from copy import deepcopy
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod
import gym
import numpy as np
import enum
import cv2
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import cm, colors
from matplotlib.colors import LinearSegmentedColormap


game_description = """Text-based %s Grid Game
________________________________

State description:
P = Player that can be moved |
# = Wall where player cannot move through |
* = Rock where player cannot move through |
. = Empty space where the player can move |
X = Death trap to avoid |
G = Goal to reach |

Action options:
l = Plan to move player left |
r = Plan to move player right |
u = Plan to move player up |
d = Plan to move player down |

Game rules:
The player has four action options l, r, u and d.
Other actions are invalid and will be ignored.
At each State the player can execute an action when queried.
If an action is not possible it will be ignored.
The game starts at State 0 on a randomly generated %s Grid world.
The player P has at most 50 states to reach the goal.

Objective:
The closer the player P gets to the goal G the higher the reward.
The optimal solution is if the player P takes the shortest path to the goal G.
If the player P falls into a trap X he dies and loses the game.
The main objective is to find optimal solution and get the highest total reward.

________________________________

"""


game_description_shifted = """Text-based %s Grid Game
________________________________

State description:
P = Player that can be moved |
# = Wall where player cannot move through |
* = Rock where player cannot move through |
T = Tree where player cannot move through |
. = Empty space where the player can move |
X = Death trap to avoid |
G = Goal to reach |

Action options:
l = Plan to move player left |
r = Plan to move player right |
u = Plan to move player up |
d = Plan to move player down |

Game rules:
The player has four action options l, r, u and d.
Other actions are invalid and will be ignored.
At each State the player can execute an action when queried.
If an action is not possible it will be ignored.
The game starts at State 0 on a randomly generated %s Grid world.
The player P has at most 50 states to reach the goal.

Objective:
The closer the player P gets to the goal G the higher the reward.
The optimal solution is if the player P takes the shortest path to the goal G.
If the player P falls into a trap X he dies and loses the game.
The main objective is to find optimal solution and get the highest total reward.

________________________________

"""

def prepare_state(state: str, grid: Tuple[int, int], add_description: bool, domain_shift: bool):
    query = ''
    if add_description and 'State 0' in state:
        descr = game_description_shifted if domain_shift else game_description
        if '>' in descr or '(' in descr:
            raise ValueError('Description contains invalid characters: > or ( are not allowed')
        descr = descr.replace('\n', '\\n')
        query += descr % (grid, grid)
    query += state
    if 'Game ended' not in query:
        query = query[:query.rfind('>')]
        query += '->'
    return query


class Position(ABC):
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y
        
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Position(x, y)
    
    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Position(x, y)
    
    def __hash__(self) -> int:
        return hash(f"({self.x}, {self.y})")
    
    def __eq__(self, other):
        return self.x == other.x and \
            self.y == other.y
            
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


class Obstacle(ABC):
    def __init__(self, pos: Position):
        self.id: str = '+'
        self.pos: Position = pos
        self.pixels: int = 7
        
    @property
    def terminal(self) -> bool:
        return False

    def manhatten_dist(self, pos: Position) -> int:
        d = Position(self.pos.x - pos.x, self.pos.y - pos.y)
        return np.abs(d.x) + np.abs(d.y)
    
    def color(self) -> Tuple[int, int, int, int]:
        return (0, 0, 0, 255)
    
    def font_color(self) -> Tuple[int, int, int, int]:
        return (255, 255, 255, 255)
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
    
    def tile(self) -> np.ndarray:
        tile = np.full((self.pixels, self.pixels, 4), self.color(), dtype=np.float32)
        tile[self.text().T == 1] = self.font_color()
        return tile
    
    def reward(self) -> float:
        return 0.0
    
    def __sub__(self, other) -> Position:
        if isinstance(other, Position):
            return Position(self.pos.x - other.x, self.pos.y - other.y)
        return Position(self.pos.x - other.pos.x, self.pos.y - other.pos.y)

    def __add__(self, other) -> Position:
        if isinstance(other, Position):
            return Position(self.pos.x + other.x, self.pos.y + other.y)
        return Position(self.pos.x + other.pos.x, self.pos.y + other.pos.y)
    
    def __str__(self) -> str:
        return self.id
    
    def __eq__(self, other):
        return self.id == other.id and \
            self.pos == other.pos


class Terminal(Obstacle):
    @property
    def terminal(self) -> bool:
        return True


class Player(Obstacle):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = 'P'
        self.action_letter = ['l', 'r', 'u', 'd']
        self.action_map = {i: self.action_letter[i] for i in range(len(self.action_letter))}
        self.inv_action_map = {self.action_letter[i]: i for i in range(len(self.action_letter))}

    def sample(self) -> str:
        return self.action_letter[np.random.randint(low=0, high=len(self.action_letter))]
    
    def is_valid_action(self, action: str):
        return action in self.action_letter
    
    def color(self) -> Tuple[int, int, int, int]:
        return (15, 82, 186, 255)
    
    def to_pos(self, val) -> Position:
        if isinstance(val, str):
            letter = val
            if letter == 'l':
                return Position(-1, 0)
            elif letter == 'r':
                return Position(+1, 0)
            elif letter == 'u':
                return Position(0, -1)
            elif letter == 'd':
                return Position(0, +1)
            else:
                raise ValueError(f'Invalid action letter: {letter}')
        elif isinstance(val, int) or isinstance(val, np.int64):
            return self.to_pos(self.action_letter[val])
        else:
            raise ValueError(f'Invalid action type: {type(val)}')
        
    def to_letter(self, val) -> str:
        if isinstance(val, Position):
            pos = val
            if Position(0, -1) == pos:
                return 'u'
            elif Position(0, +1) == pos:
                return 'd'
            elif Position(-1, 0) == pos:
                return 'l'
            elif Position(+1, 0) == pos:
                return 'r'
            else: 
                raise ValueError(f'Invalid action position: {pos}')
        elif isinstance(val, int) or isinstance(val, np.int64):
            return self.action_letter[val]
        else:
            raise ValueError(f'Invalid action type: {type(val)}')

    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 1, 1, 0, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 1, 1, 0, 0 ],
                [ 0, 0, 1, 0, 0, 0, 0 ],
                [ 0, 0, 1, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )


class Goal(Terminal):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = 'G'

    def reward(self) -> float:
        return 1.0
    
    def color(self) -> Tuple[int, int, int, int]:
        return (85, 26, 139, 255)
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 1, 1, 1, 0, 0 ],
                [ 0, 1, 0, 0, 0, 0, 0 ],
                [ 0, 1, 0, 0, 1, 1, 0 ],
                [ 0, 1, 0, 0, 0, 1, 0 ],
                [ 0, 0, 1, 1, 1, 1, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        
        
class Path(Obstacle):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = 'o'


class Empty(Obstacle):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = '.'
        
    def color(self) -> Tuple[int, int, int, int]:
        return (190, 190, 190, 255)


class Solid(Obstacle):
    pass


class Wall(Solid):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = '#'
        
    def color(self) -> Tuple[int, int, int, int]:
        return (45, 40, 45, 255)
        
        
class Tree(Solid):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = 'T'
        
    def color(self) -> Tuple[int, int, int, int]:
        return (40, 200, 0, 255)
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 1, 1, 1, 1, 1, 0 ],
                [ 0, 0, 0, 1, 0, 0, 0 ],
                [ 0, 0, 0, 1, 0, 0, 0 ],
                [ 0, 0, 0, 1, 0, 0, 0 ],
                [ 0, 0, 0, 1, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        
        
class Rock(Solid):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = '*'
    
    def color(self) -> Tuple[int, int, int, int]:
        return (121, 68, 59, 255)
        
        
class Trap(Terminal):
    def __init__(self, pos: Position):
        super().__init__(pos)
        self.id = 'X'

    def reward(self) -> float:
        return -1
    
    def color(self) -> Tuple[int, int, int, int]:
        return (140, 12, 2, 255)
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 1, 0, 0, 0, 1, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 0, 1, 0, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 1, 0, 0, 0, 1, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        
        
class ActionViewLeft(Obstacle):
    def __init__(self, pos: Position, color: Tuple[int, int, int, int]):
        super().__init__(pos)
        self.id = 'l'
        self._color = color
    
    def color(self) -> Tuple[int, int, int, int]:
        return self._color
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 1, 0, 0, 0, 0 ],
                [ 0, 0, 1, 0, 0, 0, 0 ],
                [ 0, 0, 1, 0, 0, 0, 0 ],
                [ 0, 0, 1, 0, 0, 0, 0 ],
                [ 0, 0, 1, 1, 1, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        

class ActionViewRight(Obstacle):
    def __init__(self, pos: Position, color: Tuple[int, int, int, int]):
        super().__init__(pos)
        self.id = 'r'
        self._color = color
    
    def color(self) -> Tuple[int, int, int, int]:
        return self._color
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 1, 1, 1, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 1, 0, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        
        
class ActionViewUp(Obstacle):
    def __init__(self, pos: Position, color: Tuple[int, int, int, int]):
        super().__init__(pos)
        self.id = 'u'
        self._color = color
    
    def color(self) -> Tuple[int, int, int, int]:
        return self._color
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 1, 1, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        
        
class ActionViewDown(Obstacle):
    def __init__(self, pos: Position, color: Tuple[int, int, int, int]):
        super().__init__(pos)
        self.id = 'd'
        self._color = color
    
    def color(self) -> Tuple[int, int, int, int]:
        return self._color
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 1, 1, 0, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 0, 1, 0, 0 ],
                [ 0, 0, 1, 1, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        
        
class ActionViewNoOp(Obstacle):
    def __init__(self, pos: Position, color: Tuple[int, int, int, int] = (85, 80, 85, 255)):
        super().__init__(pos)
        self.id = 'n'
        self._color = color
    
    def color(self) -> Tuple[int, int, int, int]:
        return self._color
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 1, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0 ]
            ]
        )
        
        
class PlayerTracker(Obstacle):
    def __init__(self, pos: Position, color: Tuple[int, int, int, int] = (255, 0, 0, 60)):
        super().__init__(pos)
        self.id = 't'
        self.font_color_ = color
    
    def color(self) -> Tuple[int, int, int, int]:
        return (0, 0, 0, 0)
    
    def font_color(self) -> Tuple[int, int, int, int]:
        return self.font_color_
    
    def text(self) -> np.ndarray:
        return np.array(
            [
                [ 1, 1, 1, 1, 1, 1, 1 ],
                [ 1, 0, 0, 0, 0, 0, 1 ],
                [ 1, 0, 0, 0, 0, 0, 1 ],
                [ 1, 0, 0, 0, 0, 0, 1 ],
                [ 1, 0, 0, 0, 0, 0, 1 ],
                [ 1, 0, 0, 0, 0, 0, 1 ],
                [ 1, 1, 1, 1, 1, 1, 1 ]
            ]
        )


def rand_move(sign: str) -> int:
    return np.random.randint(low=-1, high=1) if sign == '+' else np.random.randint(low=0, high=2)


class Info(enum.Enum):
    Success = 1
    Failed = 2
    Death = 3
    
    
class Grid(ABC):
    def __init__(self, size, env):
        self.env = env
        self.rows = size[0]
        self.cols = size[1]
        self.size: Tuple[int, int] = size
        self._grid: List[List[Obstacle]] = None
        self.V: Dict[Position: float] = None
        self.P = None
        self.policy = None
        self.gamma = 0.95
        self.theta = 1e-10
        self.reset()
        
    def reset(self):
        self._grid = [[Obstacle(Position(i, j)) for i in range(self.cols)] for j in range(self.rows)]
        
    def view(self, place_player: bool = True, player: Player = None) -> np.ndarray:
        img = []
        for i in range(self.rows):
            r = []
            for j in range(self.cols):
                p = player if player is not None else self.env.player
                if place_player and self._grid[i][j].pos == p.pos:
                    obj = p.tile()
                else:
                    obj = self._grid[i][j].tile()
                r.append(obj)
            img.append(np.concatenate(r, axis=1))
        img = np.concatenate(img, axis=0)
        return img.transpose(1, 0, 2)
    
    def value_function(self, state: Position) -> float:
        v = self.V.get(state, -np.inf)
        return v
    
    def action_view(self, place_player: bool = True) -> np.ndarray:
        img = []
        history = [p.pos for p in self.env.player_history]
        colors = list(Color('yellow').range_to(Color("red"), len(history)))
        
        for i in range(self.rows):
            r = []
            for j in range(self.cols):
                obj = self.policy[i][j].tile()
                if place_player and len(self.env.player_history) > 0:
                    if Position(i, j) in history:
                        c = colors[history.index(Position(i, j))]
                        c = (int(c.get_red() * 255), int(c.get_blue() * 255), int(c.get_green() * 255), 80)
                        overlay = PlayerTracker(Position(i, j), c).tile()
                        obj += overlay
                        obj = obj // 2
                r.append(obj)
            img.append(np.concatenate(r, axis=1))
        img = np.concatenate(img, axis=0)
        return img.transpose(1, 0, 2)

    def init_tables(self):
        step_reward = -0.1
        self.P = {}
        self.V = {o.pos: 0.0 for o in sum(self._grid, []) if not isinstance(o, Wall)}
        self.policy = np.full(np.prod(self.size), 'n').reshape(self.size)
        for state in sum(self._grid, []):
            if isinstance(state, Wall):
                continue
            for letter in self.env.player.action_letter:
                reward = step_reward
                action = self.env.player.to_pos(letter)
                n_pos = state + action
                n_state = self._grid[n_pos.x][n_pos.y]
                
                if isinstance(n_state, Goal):
                    reward += n_state.reward()
                elif isinstance(n_state, Trap):
                    reward += n_state.reward()
                elif isinstance(n_state, Solid):
                    n_state = state
                
                self.P[(state.pos, action)] = (n_state.pos, reward)
        
    def is_ignored(self, obst: Obstacle) -> bool:
        return isinstance(obst, Solid) or isinstance(obst, Terminal)
        
    def interate_values(self):
        converged = False
        i = 0
        # iterate values
        while not converged:
            DELTA = 0
            for state in sum(self._grid, []):
                if isinstance(state, Wall):
                    continue
                i += 1
                if  state.terminal:
                    self.V[state.pos] = 0
                else:
                    old_v = self.V[state.pos]
                    new_v = []
                    for letter in self.env.player.action_letter:
                        action = self.env.player.to_pos(letter)
                        (n_state_pos, reward) = self.P.get((state.pos, action))
                        n_state = self._grid[n_state_pos.x][n_state_pos.y]
                        new_v.append(reward + self.gamma * self.V[n_state.pos])

                    self.V[state.pos] = max(new_v)

                    DELTA = max(DELTA, np.abs(old_v - self.V[state.pos]))
                    converged = True if DELTA < self.theta else False

        # get optimal policy
        for state in sum(self._grid, []):
            if self.is_ignored(state):
                continue
            i += 1
            new_vs = []

            for letter in self.env.player.action_letter:
                action = self.env.player.to_pos(letter)
                (n_state_pos, reward) = self.P.get((state.pos, action))
                new_vs.append(reward + self.gamma * self.V[n_state_pos])

            new_vs = np.array(new_vs)
            best_action_idx = np.where(new_vs == new_vs.max())[0]
            self.policy[state.pos.x][state.pos.y] = self.env.player.to_letter(best_action_idx[0])
            
        new_policy = [[None for i in range(self.cols)] for j in range(self.rows)]
        min_v = min(self.V.values())
        max_v = max(self.V.values())
        
        def bluegreen(y):
            red = [(0.0, 0.0, 0.0), (0.5, y, y), (1.0, 0.0, 0.0)]
            green = [(0.0, 0.0, 0.0), (0.5, y, y), (1.0, y, y)]
            blue = [(0.0, y, y), (0.5, y, y), (1.0, 0.0, 0.0)]
            colordict = dict(red=red, green=green, blue=blue)
            bluegreenmap = LinearSegmentedColormap('bluegreen', colordict, 256)
            return bluegreenmap
        
        norm = colors.Normalize(vmin=min_v, vmax=max_v, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=bluegreen(0.5))
        
        for i in range(self.rows):
            for j in range(self.cols):
                p = self.policy[i][j]
                
                if p == 'u' or p == 'd' or p == 'l' or p == 'r':
                    c = mapper.to_rgba(self.V[Position(i, j)])
                    c = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255)
                
                if p == 'n':
                    new_policy[i][j] = ActionViewNoOp(Position(i, j))
                elif p == 'u':
                    new_policy[i][j] = ActionViewUp(Position(i, j), c)
                elif p == 'd':
                    new_policy[i][j] = ActionViewDown(Position(i, j), c)
                elif p == 'l':
                    new_policy[i][j] = ActionViewLeft(Position(i, j), c)
                elif p == 'r':
                    new_policy[i][j] = ActionViewRight(Position(i, j), c)
                else:
                    raise ValueError(f'Invalid policy {p}')
        
        self.policy = new_policy
        
    def __len__(self):
        return len(self._grid)
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._grid[key] = value
        elif isinstance(key, Tuple[int, int]):
            self._grid[key[0]][key[1]] = value
        elif isinstance(key, Position):
            self._grid[key.x][key.y] = value
        else:
            raise ValueError('Unsupported index type')
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._grid[key]
        elif isinstance(key, Tuple):
            return self._grid[key[0]][key[1]]
        elif isinstance(key, Position):
            return self._grid[key.x][key.y]
        else:
            raise ValueError('Unsupported index type')


class GridEnv(gym.Env):
    def __init__(self, 
                 size: Position, 
                 trap_prob: float = 0.05,
                 obst_prob: float = 0.2,
                 tree_prob: float = 0.0, 
                 additional_horizon: int = 30,
                 add_description: bool = False,
                 add_value_function: bool = False,
                 render_image: bool = False,
                 debug: bool = False,
                 clean_single_solids: bool = True):
        self.debug = debug
        self.clean_single_solids = clean_single_solids
        self.render_image = render_image
        self.rows = size[0]
        self.cols = size[1]
        self.size = (size[0], size[1])
        self.action_space = None
        self.player: Player = None
        self.player_history: List[Player] = []
        self.action_history: List[str] = []
        self._grid: Grid = None
        self.goal: Goal = None
        self.timestep: int = 0
        self.total_reward: float = 0.0
        self.state: str = ''
        self.path: List[Position] = []
        self.done: bool = False
        self.init: bool = False
        self.frames: List[np.ndarray] = None
        self.success: Info = Info.Failed
        self.trap_p: float = trap_prob
        self.obst_p: float = obst_prob
        self.tree_p: float = tree_prob
        self.max_horizon: int = max(self.rows + self.cols + additional_horizon - 1, 50)
        self.add_description: bool = add_description
        self.add_value_function: bool = add_value_function

    def rand_pos(self, axis: str) -> int:
        return np.random.randint(low=1, high=self.rows-1) if axis == 'x' \
            else np.random.randint(low=1, high=self.cols-1)
            
    def generate_path(self) -> List[Position]:
        res = [self.goal.pos]
        d = self.goal - self.player
        t_move_x = 0
        t_move_y = 0
        
        while d.x != 0 or d.y != 0:
            move_x = 0
            if d.x > 0:
                move_x = rand_move('+')
            elif d.x < 0:
                move_x = rand_move('-')

            move_y = 0
            if move_x == 0:
                if d.y > 0:
                    move_y = -1
                elif d.y < 0:
                    move_y = +1
                    
            t_move_x += move_x
            t_move_y += move_y
            
            d = Position(d.x + move_x, d.y + move_y)
            obj = Position(self.goal.pos.x + t_move_x, self.goal.pos.y + t_move_y)
            if obj == self.player.pos:
                break
            if obj != res[-1]:
                res.append(obj)
            
        res.append(self.player.pos)
        assert res[0] == self.goal.pos and res[-1] == self.player.pos
        self.path = res

    def reset(self, *args, **kwargs) -> str:
        self.frames = []
        self.success = Info.Failed
        self.total_reward = 0.0
        self._grid = Grid(self.size, self)
        g_pos_x = self.rand_pos('x')
        g_pos_y = self.rand_pos('y')
        self.goal = Goal(Position(g_pos_x, g_pos_y))
        p_pos_x = self.rand_pos('x')
        p_pos_y = self.rand_pos('y')
        self.player = Player(Position(p_pos_x, p_pos_y))
        self.action_space = self.player.action_letter
        retry = 0
        while g_pos_x == p_pos_x and g_pos_y == p_pos_y or self.goal.manhatten_dist(self.player.pos) < min(self.rows, self.cols):
            p_pos_x = self.rand_pos('x')
            p_pos_y = self.rand_pos('y')
            self.player = Player(Position(p_pos_x, p_pos_y))
            if retry > 10:
                g_pos_x = self.rand_pos('x')
                g_pos_y = self.rand_pos('y')
                self.goal = Goal(Position(g_pos_x, g_pos_y))
            if retry > 20 and not (g_pos_x == p_pos_x and g_pos_y == p_pos_y):
                break
            retry += 1
        self.player_history = [deepcopy(self.player)]
        self.action_history = []
        
        # generate optimal path
        self.generate_path()
        
        # create base walls
        self._grid[0] = [Wall(Position(0, i)) for i in range(self.cols)]
        self._grid[-1] = [Wall(Position(0, i)) for i in range(self.cols)]
        for i in range(1, self.rows):
            self._grid[i][0] = Wall(Position(i, 0))
            self._grid[i][-1] = Wall(Position(i, self.rows-1))

        # randomly add some walls if not on path
        for i in range(1, self.rows-1):
            for j in range(1, self.cols-1):
                if self._grid[i][j].pos in self.path:
                    self._grid[i][j] = Empty(Position(i, j))
                else:
                    # random object placement
                    p = np.random.uniform(low=0, high=1)
                    if p > 1-self.trap_p:
                        # place death
                        self._grid[i][j] = Trap(Position(i, j))
                    else:
                        # add rock or wall with probability
                        p = np.random.uniform(low=0, high=1)
                        class_ = Tree if p < self.tree_p else Rock
                        # place wall with probability
                        p = np.random.uniform(low=0, high=1)
                        self._grid[i][j] = class_(Position(i, j)) if p < self.obst_p else Empty(Position(i, j))
        
        # cleanup alone standing walls
        if self.clean_single_solids:
            for i in range(1, self.rows-1):
                for j in range(1, self.cols-1):
                    is_alone = True
                    obj = self._grid[i][j]
                    if isinstance(obj, Solid):
                        if isinstance(self._grid[obj + Position(1, 0)], Solid) or \
                            isinstance(self._grid[obj + Position(-1, 0)], Solid) or \
                                isinstance(self._grid[obj + Position(0, 1)], Solid) or \
                                    isinstance(self._grid[obj + Position(0, -1)], Solid):
                                        is_alone = False
                        if is_alone:
                            self._grid[i][j] = Empty(obj.pos)
              
        # show optimal path  
        for p in self.path:
            self._grid[p.x][p.y] = Empty(p)
        self._grid[g_pos_x][g_pos_y] = self.goal
        
        self.timestep = 0
        self.done = False
        self._grid.init_tables()
        self._grid.interate_values()
        self.state = self.render()
        self.init = True
        return self.state
    
    def validate_action(self, action: str):
        if not self.player.is_valid_action(action):
            return self.sample(), False
        return action, True
    
    def sample(self) -> str:
        return self.player.sample()
    
    def optimal(self) -> str:
        if self.timestep >= len(self.path): ValueError('Too many steps')
        pos = self.path[-self.timestep-1]
        next_pos = self.path[-self.timestep-2]
        
        move = next_pos - pos
        if move == Position(-1, 0):
            return self.player.action_letter[0]
        elif move == Position(1, 0):
            return self.player.action_letter[1]
        elif move == Position(0, -1):
            return self.player.action_letter[2]
        elif move == Position(0, 1):
            return self.player.action_letter[3]
        else:
            raise ValueError('Unknown Command')
        
    def prepare_state(self, state: str, grid: Tuple[int, int], add_description: bool, domain_shift: bool):
        return prepare_state(state, grid, add_description, domain_shift)

    def render(self) -> str:
        reward = 0.0
        res = '%s Grid:\\n' % (self.size,)
        res += f'State {self.timestep},\\n'
        for j in range(self.cols):
            for i in range(self.rows):
                if self.player.pos == Position(i, j):
                    res += self.player.id
                else:
                    res += self._grid[i][j].id
            if j < self.cols - 1:
                res += '\\n'
        res += ','
        if self.add_description:
            dist = self.player.manhatten_dist(self.goal.pos)
            res += f'\\nGoal Distance {dist},'
            res += f'\\nReward {reward:.0f},'
            if self.add_value_function:
                for action in self.player.action_letter:
                    move = self.player.to_pos(action)
                    new_pos = self.player.pos + move
                    if isinstance(self._grid[new_pos], Goal):
                        value = self._grid[new_pos].reward()
                    elif isinstance(self._grid[new_pos], Trap):
                        value = self._grid[new_pos].reward()
                    else:
                        value = self._grid.value_function(new_pos)
                    res += f'\\nAction: {action} Value: {value:.2f},'
            
        if not self.done:
            res += f' > '
        else:
            if self.success == Info.Success:
                succ_message = 'Congrats, you found the goal' 
            elif self.success == Info.Death:
                succ_message = 'You died'
            else:
                succ_message = 'You failed to find the goal'
            res += f'\\nGame ended - {succ_message}! Total Timesteps: {self.timestep}, Total Reward: {self.total_reward:.0f},\\n'

        if self.render_image:
            img, _ = self.get_image()
            self.frames.append(img)
        
        return res
    
    def get_image(self) -> np.ndarray:
        grid = self._grid.view()
        action_grid = self._grid.action_view(place_player=True)
        img_grid = np.concatenate((grid, action_grid), axis=0)
        img_grid = img_grid.astype(np.uint8)
        img = Image.fromarray(img_grid, 'RGBA')
        return img, grid
    
    def save_action_view(self, output_filename: str) -> np.ndarray:
        grid = self._grid.view(place_player=True, player=self.player_history[0])
        action_grid = self._grid.action_view(place_player=False)
        img_grid = np.concatenate((grid, action_grid), axis=0)
        img_grid = img_grid.astype(np.uint8)
        img = Image.fromarray(img_grid, 'RGBA')
        
        i = 0
        os.makedirs('results', exist_ok=True)
        file_name = f'results/action_view_grid_{self.rows}x{self.cols}_{output_filename}_{i}.png'
        while os.path.exists(file_name):
            i += 1
            file_name = f'results/action_view_grid_{self.rows}x{self.cols}_{output_filename}_{i}.png'
        img.save(file_name)
        return file_name
    
    def write_video(self, output_filename: str, frame_rate: int = 3):
        if len(self.frames) == 0:
            raise ValueError('No frames to write! Verify if the render_image flag is set to True! User --video as argument to render the video.')
        os.makedirs('results/video', exist_ok=True)
        frame_size = (self.frames[0].width, self.frames[0].height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        i = 0
        file_name = f'results/video/grid_{self.rows}x{self.cols}_{output_filename}_{i}.mp4'
        while os.path.exists(file_name):
            i += 1
            file_name = f'results/video/grid_{self.rows}x{self.cols}_{output_filename}_{i}.mp4'
        out = cv2.VideoWriter(file_name, 
                              fourcc, frame_rate, frame_size)
        for idx in range(len(self.frames)*frame_rate):
            frame = self.frames[idx//frame_rate]
            open_cv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGR)
            out.write(open_cv_image)
        out.release()
        return file_name

    def step(self, action: str) -> Tuple[str, bool, dict]:
        if not self.init: raise ValueError('Called Step before initialization!')
        if self.done: raise ValueError('Game is done. Please reset before calling step(..)!')
        if self.timestep >= self.max_horizon: self.done = True
        self.action_history.append(action)

        reward = 0.0
        move = Position(0, 0)
        if action == self.player.action_letter[0]:
            move = Position(-1, 0)
        elif action == self.player.action_letter[1]:
            move = Position(1, 0)
        elif action == self.player.action_letter[2]:
            move = Position(0, -1)
        elif action == self.player.action_letter[3]:
            move = Position(0, 1)
        else:
            if self.debug: print(f'Unknown Command: {action}')
        
        self.state = f'{action};\\n\\n'
        self.timestep += 1
        
        # verify if accept command
        new_pos = self.player + move
        if new_pos.x < 0 or new_pos.y < 0 or \
                new_pos.x >= self.rows or new_pos.y >= self.cols:
            self.state += self.render()
            self.player_history.append(deepcopy(self.player))
            return self.state, reward, self.done, {}
        if isinstance(self._grid[new_pos], Solid):
            self.state += self.render()
            self.player_history.append(deepcopy(self.player))
            return self.state, reward, self.done, {}
        
        # verify if game finished
        if isinstance(self._grid[new_pos], Goal):
            self.success = Info.Success
            self.done = True
            reward = 1
            self.total_reward += reward
        if isinstance(self._grid[new_pos], Trap):
            self.success = Info.Death
            self.done = True
            reward = -1
            self.total_reward += reward
        
        new_player = Player(new_pos)
        # verify if player moves towards goal
        proximity = max(self.player.manhatten_dist(self.goal.pos) - new_player.manhatten_dist(self.goal.pos), 0)

        self.player = new_player
        self.player_history.append(deepcopy(self.player))
        self.state += self.render()
        return self.state, reward, self.done, {'proximity': proximity}


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--add_description', action=argparse.BooleanOptionalAction,
                        help='add description to starting state')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction,
                        help='add debug flag')
    parser.add_argument('--domain_shift', action=argparse.BooleanOptionalAction,
                        help='add domain shift to description')
    parser.add_argument('--grid_size', type=Tuple[int, int],
                        default=(13, 9), help='size of the grid')
    args = parser.parse_args()       
    return args


def manual_game(args):    
    env = GridEnv(args.grid_size, 
                  add_description=args.add_description,
                  render_image=True,
                  tree_prob=0.5,
                  debug=True)
    
    state = env.reset()
    episode = state
    done = False
    while not done:
        prep_state = env.prepare_state(state, args.grid_size, args.add_description, args.domain_shift)
        print(prep_state.replace('\\n', '\n'))
        action = input('')
        state, reward, done, _ = env.step(action)
        episode += state
    env.write_video('manual_game')


if __name__ == "__main__":
    args = parse_args()
    manual_game(args)
