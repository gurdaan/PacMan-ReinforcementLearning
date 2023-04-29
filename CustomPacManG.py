import gym
from gym import spaces
import pygame
import numpy as np
import tcod
import random
from enum import Enum
import numpy as np
from PIL import ImageGrab
import tensorflow as tf
import time

class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    sess = tf.compat.v1.Session()

    def __init__(self):
        tf.compat.v1.disable_eager_execution()

        # set up a TensorFlow session
        
        # set up a placeholder for the observation
        self.obs_placeholder = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None, None, 3))

        # preprocess the observation
        self.obs_resized = tf.image.resize(self.obs_placeholder, [84, 84])
        self.obs_grayscale = tf.image.rgb_to_grayscale(self.obs_resized)
        self.obs_normalized = tf.divide(self.obs_grayscale, 255.0)
        # set up a TensorFlow session
        
        # np.set_printoptions(threshold=np.inf)
        # img = ImageGrab.grab()
        #     # Convert the image to a 3D array of RGB pixel values
        # self.obs = np.array(img)
        # # self.obs = obs.transpose((2, 0, 1))
        # sess = tf.compat.v1.Session()
        # self.obs_processed= sess.run(self.obs_normalized, feed_dict={self.obs_placeholder: self.obs})


        super().__init__()

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: LEFT, 1: UP, 2: RIGHT, 3: DOWN
        # self.observation_space = spaces.Box(low=-500, high=500, shape=(3, 2), dtype=np.float32)
        #self.observation_space = spaces.Box(low=-500, high=500, shape=(3, 2), dtype=np.float32)
        num_positions = 3
        # obs_low = np.array([(-np.inf, -np.inf), (-np.inf, -np.inf), (-np.inf, -np.inf)])  # minimum value for each dimension
        # obs_high = np.array([(np.inf, np.inf), (np.inf, np.inf), (np.inf, np.inf)])  # maximum value for each dimension
        # obs_shape = (num_positions, 2)  # shape of the observation space
        # print(spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float64))
        # self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def reset(self):
        #observation = self._get_observation()
        self.done=False
      
        unified_size = 20
        self.pacman_game = PacmanGameController()
        size = self.pacman_game.size
        self.game_renderer = GameRenderer(size[0] * unified_size, size[1] * unified_size)
        self.game_renderer._done=False

        for y, row in enumerate(self.pacman_game.numpy_maze):
            for x, column in enumerate(row):
                if column == 0:
                    self.game_renderer.add_wall(Wall(self.game_renderer, x, y, unified_size))

        # for cookie_space in self.pacman_game.cookie_spaces:
        #     translated = translate_maze_to_screen(cookie_space)
        #     self.cookie = Cookie(self.game_renderer, translated[0] + unified_size / 2, translated[1] + unified_size / 2)
        #     self.game_renderer.add_cookie(self.cookie)

        for cookie_space in self.pacman_game.cookie_spaces:
            x, y = cookie_space
            x *= unified_size
            y *= unified_size
            translated = translate_maze_to_screen(cookie_space)
            cookie = Cookie(self.game_renderer, x + unified_size / 2, y + unified_size / 2)
            self.game_renderer.add_cookie(cookie)

        self.pacman = Hero(self.game_renderer, unified_size, unified_size, unified_size)

        self.game_renderer.add_hero(self.pacman)

        for i, ghost_spawn in enumerate(self.pacman_game.ghost_spawns):
            translated = translate_maze_to_screen(ghost_spawn)
            self.ghost = Ghost(self.game_renderer, translated[0], translated[1], unified_size, self.pacman_game,self.pacman,self.game_renderer,
                        self.pacman_game.ghost_colors[i % 4])
            self.game_renderer.add_game_object(self.ghost)

        # create observation:
        img = ImageGrab.grab()
        # Convert the image to a 3D array of RGB pixel values
        self.obs = np.array(img)
        # self.obs = obs.transpose((2, 0, 1))
        # sess = tf.compat.v1.Session()
        self.obs_processed= self.sess.run(self.obs_normalized, feed_dict={self.obs_placeholder: self.obs})
        
        # observation = self.obs
        # observation = np.array(observation)
        # print((self.obs), end="\n\n")
        # print((self.observation_space))
        return np.transpose(self.obs_processed, (2, 0, 1))
        # return self.obs_processed

    def step(self, action):        
        black = (0, 0, 0)

        if action==0:
            self.pacman.set_direction(Direction.UP)
        elif action==1:
            self.pacman.set_direction(Direction.LEFT)
        elif action==2:
            self.pacman.set_direction(Direction.DOWN)    
        else:
            self.pacman.set_direction(Direction.RIGHT)

        score_text = self.game_renderer.score_font.render("Score: " + str(self.pacman.rewards), True, (255, 255, 255))
        text_rect = score_text.get_rect()
        text_rect.left = 10  # Set the left position of the text rectangle
        text_rect.top = 450   # Set the top position of the text rectangle
        self.game_renderer._screen.blit(score_text, text_rect)
        

        for game_obj in self.game_renderer._game_objects:
            game_obj.tick()
            game_obj.draw()
        

        if self.game_renderer._done:
            self.done=True

        img = ImageGrab.grab()
            # Convert the image to a 3D array of RGB pixel values
        self.obs = np.array(img)
        # self.obs = obs.transpose((2, 0, 1))
        #sess = tf.compat.v1.Session()
        self.obs_processed= self.sess.run(self.obs_normalized, feed_dict={self.obs_placeholder: self.obs})

        pygame.display.flip()

        self.game_renderer._clock.tick(20)
        self.game_renderer._screen.fill(black)
        

        # Update the game state and get the observation and reward
        #self._game_renderer.tick(30,self.pacman,self.game_renderer)
        #observation = self._get_observation()
        reward = self.pacman.rewards

        # Check if the game is over
        # if self.pacman.rewards >= 2:
        #     self.done = True
        # for obj in self.game_renderer.get_game_objects():
        #     if isinstance(obj, Ghost):
        # # do something if the object is of class Ghosts
        #         if((self.pacman.get_position()[0]==obj.ghost.get_position()[0]) and (self.pacman.get_position()[1]==obj.ghost.get_position()[1])):
        #             print("Collided")
        #             self.pacman.rewards=-10
        #           self.done=True
        

        # create observation:
        # observation = self.obs
        # observation = np.array(observation)
        # print(observation.dtype)
        # print("Observation :",self.obs)

        return np.transpose(self.obs_processed,(2,0,1)), reward, self.done, {}
    # def _get_observation(self):
    #         # return current observation, expanded to 3 channels
    #         return np.concatenate([self._observation]*3, axis=0)
class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    NONE = 4

def translate_screen_to_maze(in_coords, in_size=20):
    return int(in_coords[0] / in_size), int(in_coords[1] / in_size)

def translate_maze_to_screen(in_coords, in_size=20):
    return in_coords[0] * in_size, in_coords[1] * in_size

class GameObject:
    def __init__(self, in_surface, x, y,
                 in_size: int, in_color=(255, 0, 0),
                 is_circle: bool = False):
        self._size = in_size
        self._renderer: GameRenderer = in_surface
        self._surface = in_surface._screen
        self.y = y
        self.x = x
        self._color = in_color
        self._circle = is_circle
        self._shape = pygame.Rect(self.x, self.y, in_size, in_size)

    def draw(self):
        if self._circle:
            pygame.draw.circle(self._surface,
                               self._color,
                               (self.x, self.y),
                               self._size)
        else:
            rect_object = pygame.Rect(self.x, self.y, self._size, self._size)
            pygame.draw.rect(self._surface,
                             self._color,
                             rect_object,
                             border_radius=4)

    def tick(self):
        pass

    def get_shape(self):
        return self._shape

    def set_position(self, in_x, in_y):
        self.x = in_x
        self.y = in_y

    def get_position(self):
        return (self.x, self.y)

class Wall(GameObject):
    def __init__(self, in_surface, x, y, in_size: int, in_color=(0, 0, 255)):
        super().__init__(in_surface, x * in_size, y * in_size, in_size, in_color)

class GameRenderer:
    def __init__(self, in_width: int, in_height: int):
        pygame.init()
        self._width = in_width
        self._height = in_height
        self._screen = pygame.display.set_mode((560, 500))
        pygame.display.set_caption('Pacman')
        self._clock = pygame.time.Clock()
        self._done = False
        self._game_objects = []
        self._walls = []
        self._cookies = []
        self._hero: Hero = None
        self.score_font = pygame.font.Font(None, 36)

    # def tick(self, in_fps: int,pacman,game_renderer):
    #     black = (0, 0, 0)
    #     while not self._done:

            
    #         score_text = self.score_font.render("Score: " + str(pacman.rewards), True, (255, 255, 255))
    #         text_rect = score_text.get_rect()
    #         text_rect.left = 10  # Set the left position of the text rectangle
    #         text_rect.top = 10   # Set the top position of the text rectangle
    #         game_renderer._screen.blit(score_text, text_rect)
            

    #         for game_object in self._game_objects:
    #             game_object.tick()
    #             game_object.draw()

    #         pygame.display.flip()
    #         self._clock.tick(in_fps)
    #         self._screen.fill(black)
    #         self._handle_events()
    #     print("Game over")

    def add_game_object(self, obj: GameObject):
        self._game_objects.append(obj)

    def add_cookie(self, obj: GameObject):
        self._game_objects.append(obj)
        self._cookies.append(obj)

    def add_wall(self, obj: Wall):
        self.add_game_object(obj)
        self._walls.append(obj)

    def get_walls(self):
        return self._walls

    def get_cookies(self):
        return self._cookies

    def get_game_objects(self):
        return self._game_objects

    def add_hero(self, in_hero):
        self.add_game_object(in_hero)
        self._hero = in_hero

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._done = True

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            self._hero.set_direction(Direction.UP)
        elif pressed[pygame.K_LEFT]:
            self._hero.set_direction(Direction.LEFT)
        elif pressed[pygame.K_DOWN]:
            self._hero.set_direction(Direction.DOWN)
        elif pressed[pygame.K_RIGHT]:
            self._hero.set_direction(Direction.RIGHT)

class MovableObject(GameObject):
    def __init__(self, in_surface, x, y, in_size: int, in_color=(255, 0, 0), is_circle: bool = False):
        super().__init__(in_surface, x, y, in_size, in_color, is_circle)
        self.current_direction = Direction.NONE
        self.direction_buffer = Direction.NONE
        self.last_working_direction = Direction.NONE
        self.location_queue = []
        self.next_target = None
        self.rewards=0
        self.pacman_pos=()

    def get_next_location(self):
        return None if len(self.location_queue) == 0 else self.location_queue.pop(0)

    def set_direction(self, in_direction):
        self.current_direction = in_direction
        self.direction_buffer = in_direction

    def collides_with_wall(self, in_position):
        collision_rect = pygame.Rect(in_position[0], in_position[1], self._size, self._size)
        collides = False
        walls = self._renderer.get_walls()
        for wall in walls:
            collides = collision_rect.colliderect(wall.get_shape())
            if collides: break
        return collides

    def check_collision_in_direction(self, in_direction: Direction):
        desired_position = (0, 0)
        if in_direction == Direction.NONE: return False, desired_position
        if in_direction == Direction.UP:
            desired_position = (self.x, self.y - 1)
        elif in_direction == Direction.DOWN:
            desired_position = (self.x, self.y + 1)
        elif in_direction == Direction.LEFT:
            desired_position = (self.x - 1, self.y)
        elif in_direction == Direction.RIGHT:
            desired_position = (self.x + 1, self.y)

        return self.collides_with_wall(desired_position), desired_position

    def automatic_move(self, in_direction: Direction):
        pass

    def tick(self):
        self.reached_target()
        self.automatic_move(self.current_direction)

    def reached_target(self):
        pass

class Hero(MovableObject):
    def __init__(self, in_surface, x, y, in_size: int):
        super().__init__(in_surface, x, y, in_size, (255, 255, 0), False)
        self.last_non_colliding_position = (0, 0)
        self.start_time = time.time()
        self.last_reward=0
        self.consecutive_no_reward=0
        self.last_dist=0
        self.visited=[]
        

    def tick(self):
        # TELEPORT
        if self.x < 0:
            self.x = self._renderer._width

        if self.x > self._renderer._width:
            self.x = 0

        if not self.check_collision_in_direction(self.direction_buffer)[0]:
            self.last_non_colliding_position = (self.x, self.y)

        self.last_non_colliding_position = self.get_position()
        #print("Pacman : ",self.get_position())
        self.pacman_pos=self.get_position()
        # print("Pacman : ",self.pacman_pos)

        if self.check_collision_in_direction(self.direction_buffer)[0]:
            self.automatic_move(self.current_direction)
        else:
            self.automatic_move(self.direction_buffer)
            self.current_direction = self.direction_buffer

        if self.collides_with_wall((self.x, self.y)):
            self.set_position(self.last_non_colliding_position[0], self.last_non_colliding_position[1])

        self.handle_cookie_pickup()
        self.get_reward()
    
    def get_reward(self):
        # Calculate the Euclidean distance between Pacman's current position and its last non-colliding position
        dist = ((self.x - self.last_non_colliding_position[0])**2 + (self.y - self.last_non_colliding_position[1])**2)**0.5

        # Check if the current location has been visited before
        # current_pos = (self.x, self.y)
        # if current_pos not in self.visited:
        #     # Add a bonus reward for exploring new areas
        #     self.rewards += 10
        #     self.visited.append(current_pos)
        # else:
        #     self.rewards-=1

        # Keep track of the number of consecutive time steps where the reward has not increased
        if self.rewards == self.last_reward:
            self.consecutive_no_reward += 1
        else:
            self.consecutive_no_reward = 0
        
        self.last_reward=self.rewards

        # Penalize if there has been no reward increase for a certain number of time steps
        if self.consecutive_no_reward >= 10:
            self.rewards -= 12

        # Add a positive reward for moving away from last non-colliding position and towards unexplored areas
        reward = dist - self.last_dist
        self.rewards += reward
        self.last_dist = dist

            # Implement epsilon-greedy exploration bonus
        if random.uniform(0, 1) < 0.1:
            # Check if the current location has been visited before
            current_pos = (self.x, self.y)
            if current_pos not in self.visited:
                # Add a bonus reward for exploring new areas
                self.rewards += 10
                self.visited.append(current_pos)

        self.last_dist = dist

        # Add an additional reward for every 10 points earned
        if self.rewards % 10 == 0 and self.rewards > 0:
            self.rewards += 25

    def automatic_move(self, in_direction: Direction):
        collision_result = self.check_collision_in_direction(in_direction)

        desired_position_collides = collision_result[0]
        if not desired_position_collides:
            self.last_working_direction = self.current_direction
            desired_position = collision_result[1]
            self.set_position(desired_position[0], desired_position[1])
        else:
            self.current_direction = self.last_working_direction

    def handle_cookie_pickup(self):
        collision_rect = pygame.Rect(self.x, self.y, self._size, self._size)
        cookies = self._renderer.get_cookies()
        game_objects = self._renderer.get_game_objects()
        for cookie in cookies:
            collides = collision_rect.colliderect(cookie.get_shape())
            if collides and cookie in game_objects:
                game_objects.remove(cookie)
                self.rewards+=50
                
        

    def draw(self):
        half_size = self._size / 2
        pygame.draw.circle(self._surface, self._color, (self.x + half_size, self.y + half_size), half_size)

class Ghost(MovableObject):

    def tick(self):
        super().tick()
        size = (20, 20)
       
        ghost_rect = pygame.Rect(self.get_position(), size)
        # Get the position and dimensions of the Pacman object
        pacman_rect = pygame.Rect(self.pacman.get_position(), size)
        # Check for collision between the Ghost and Pacman objects
        if ghost_rect.colliderect(pacman_rect):
            # print("Ghost collided with Pacman!")
            self.rewards-=200
            self.game_renderer._done=True

    def __init__(self, in_surface, x, y, in_size: int, in_game_controller,pacman,game_renderer, in_color=(255, 0, 0)):
        super().__init__(in_surface, x, y, in_size, in_color, False)
        self.game_controller = in_game_controller
        self.pacman=pacman
        self.game_renderer=game_renderer

    def reached_target(self):
        if (self.x, self.y) == self.next_target:
            self.next_target = self.get_next_location()
        self.current_direction = self.calculate_direction_to_next_target()

    def set_new_path(self, in_path):
        for item in in_path:
            self.location_queue.append(item)
        self.next_target = self.get_next_location()

    def calculate_direction_to_next_target(self) -> Direction:
        if self.next_target is None:
            self.game_controller.request_new_random_path(self,self.pacman)
            return Direction.NONE
        diff_x = self.next_target[0] - self.x
        diff_y = self.next_target[1] - self.y
        if diff_x == 0:
            return Direction.DOWN if diff_y > 0 else Direction.UP
        if diff_y == 0:
            return Direction.LEFT if diff_x < 0 else Direction.RIGHT
        self.game_controller.request_new_random_path(self,self.pacman)
        return Direction.NONE

    def automatic_move(self, in_direction: Direction):
        if in_direction == Direction.UP:
            self.set_position(self.x, self.y - 1)
        elif in_direction == Direction.DOWN:
            self.set_position(self.x, self.y + 1)
        elif in_direction == Direction.LEFT:
            self.set_position(self.x - 1, self.y)
        elif in_direction == Direction.RIGHT:
            self.set_position(self.x + 1, self.y)

class Cookie(GameObject):
    def __init__(self, in_surface, x, y):
        super().__init__(in_surface, x, y, 4, (255, 255, 0), True)

class Pathfinder:
    def __init__(self, in_arr):
        cost = np.array(in_arr, dtype=np.bool_).tolist()
        self.pf = tcod.path.AStar(cost=cost, diagonal=0)

    def get_path(self, from_x, from_y, to_x, to_y) -> object:
        res = self.pf.get_path(from_x, from_y, to_x, to_y)
        return [(sub[1], sub[0]) for sub in res]

class PacmanGameController:
    def __init__(self):
        self.ascii_maze = [
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "XP           XX            X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X                          X",
            "X XXXX XX XXXXXXXX XX XXXX X",
            "X XXXX XX XXXXXXXX XX XXXX X",
            "X      XX    XX    XX      X",
            "XXXXXX XXXXX XX XXXXX XXXXXX",
            "XXXXXX XXXXX XX XXXXX XXXXXX",
            "XXXXXX XX          XX XXXXXX",
            "XXXXXX XX XXXXXXXX XX XXXXXX",
            "XXXXXX XX X   G  X XX XXXXXX",
            "            G               ",
            "XXXXXX XX X   G  X XX XXXXXX",
            "XXXXXX XX XXXXXXXX XX XXXXXX",
            "XXXXXX XX          XX XXXXXX",
            "XXXXXX XX XXXXXXXX XX XXXXXX",
            "XXXXXX XX XXXXXXXX XX XXXXXX",
            "X                          X",
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            # "X   XX       G        XX   X",
            # "XXX XX XX XXXXXXXX XX XX XXX",
            # "XXX XX XX XXXXXXXX XX XX XXX",
            # "X      XX    XX    XX      X",
            # "X XXXXXXXXXX XX XXXXXXXXXX X",
            # "X XXXXXXXXXX XX XXXXXXXXXX X",
            # "X                          X",
            # "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        ]

        self.numpy_maze = []
        self.cookie_spaces = []
        self.reachable_spaces = []
        self.ghost_spawns = []
        self.ghost_colors = [
            (255, 184, 255),
            (255, 0, 20),
            (0, 255, 255),
            (255, 184, 82)
        ]
        self.size = (0, 0)
        self.convert_maze_to_numpy()
        self.p = Pathfinder(self.numpy_maze)
        self.pacman=PacmanEnv

    def request_new_random_path(self, in_ghost: Ghost,pacman:Hero):
        RED = [(255, 184, 255),(255, 0, 20),(0, 255, 255)]
        if in_ghost._color == random.choice(RED):
            target_coord = pacman.get_position()
            random_space = translate_screen_to_maze(target_coord)
        else:
            random_space = random.choice(self.reachable_spaces)
        current_maze_coord = translate_screen_to_maze(in_ghost.get_position())

        path = self.p.get_path(current_maze_coord[1], current_maze_coord[0], random_space[1],
                            random_space[0])
        test_path = [translate_maze_to_screen(item) for item in path]
        in_ghost.set_new_path(test_path)

    def convert_maze_to_numpy(self):
        for x, row in enumerate(self.ascii_maze):
            self.size = (len(row), x + 1)
            binary_row = []
            for y, column in enumerate(row):
                if column == "G":
                    self.ghost_spawns.append((y, x))

                if column == "X":
                    binary_row.append(0)
                else:
                    binary_row.append(1)
                    self.cookie_spaces.append((y, x))
                    self.reachable_spaces.append((y, x))
            self.numpy_maze.append(binary_row)
        # print(self.numpy_maze)