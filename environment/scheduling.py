from random import randint
import pygame
import numpy as np
import time


class Work(object):
    def __init__(self, work_id=None, block=None, lead_time=1, max_days=4):
        self.id = str(work_id)
        self.block = block
        if lead_time == -1:
            self.lead_time = randint(1, 1 + max_days // 3)
        else:
            self.lead_time = lead_time


class Scheduling(object):
    def __init__(self, num_days=4, num_blocks=0, inbound_works=None, display_env=False):
        self.action_space = 3
        self.num_days = num_days
        self.num_work = len(inbound_works)
        self.num_block = num_blocks
        self.empty = 0
        self.step = 0
        self.inbound_works = inbound_works
        self.inbound_clone = inbound_works[:]
        self.works = [0]
        self._ongoing = 0
        self._location = 0
        # self.yard = np.full([max_stack, num_pile], self.empty)
        if display_env:
            display = LocatingDisplay(self, num_days, self.num_block)
            display.game_loop_from_space()

    def action(self, action):
        done = False
        reward = 0
        self.step += 1
        if action == 2:
            self._ongoing += 1
            self._location = 0
            if self._ongoing == self.num_work:
                done = True
            else:
                self.works.append(self._location)
        else:
            if action == 0:
                self._location = max(0, self._location - 1)
            elif action == 1:
                self._location = min(self.num_days - 1, self._location + 1)
            if len(self.works) == self._ongoing:
                self.works.append(self._location)
            else:
                self.works[self._ongoing] = self._location
        next_state = self.get_state()
        return next_state, reward, done

    def reset(self):
        self.inbound_works = self.inbound_clone[:]
        self.works = []
        self.step = 0
        self._ongoing = 0
        self._location = 0

    def get_state(self):
        state = np.full([self.num_work, self.num_days], 0)
        moving = 1
        confirmed = 2
        cell = 0
        for i, location in enumerate(self.works):
            if self._ongoing == i:
                cell = moving
            else:
                cell = confirmed
            for j in range(self.inbound_works[i].lead_time):
                state[self.inbound_works[i].block, location + j] = cell
        return state


class LocatingDisplay(object):
    white = (255, 255, 255)
    black = (0, 0, 0)

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    dark_red = (200, 0, 0)
    dark_green = (0, 200, 0)
    dark_blue = (0, 0, 200)

    x_init = 100
    y_init = 100
    x_span = 100
    y_span = 100
    thickness = 5
    pygame.init()
    display_width = 1000
    display_height = 600
    font = 'freesansbold.ttf'
    pygame.display.set_caption('Steel Locating')
    clock = pygame.time.Clock()
    pygame.key.set_repeat()

    def __init__(self, locating, width, height):
        self.width = width
        self.height = height
        self.space = locating
        self.on_button = False
        self.total = 0
        self.display_width = self.x_span * width + 200
        self.display_height = self.y_span * height + 200
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))

    def restart(self):
        self.space.reset()
        self.game_loop_from_space()

    def text_objects(self, text, font):
        text_surface = font.render(text, True, self.white)
        return text_surface, text_surface.get_rect()

    def block(self, x, y, text='', color=(0, 255, 0), x_init=100):
        pygame.draw.rect(self.gameDisplay, color, (int(x_init + self.x_span * x),
                                                   int(self.y_init + self.y_span * y),
                                                   int(self.x_span),
                                                   int(self.y_span)))
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(text, large_text)
        text_rect.center = (int(x_init + self.x_span * (x + 0.5)), int(self.y_init + self.y_span * (y + 0.5)))
        self.gameDisplay.blit(text_surf, text_rect)

    def board(self, step, reward=0):
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects('step: ' + str(step) + '   reward: ' + str(reward)
                                                 + '   total: ' + str(self.total), large_text)
        text_rect.center = (200, 20)
        self.gameDisplay.blit(text_surf, text_rect)

    def button(self, goal=0):
        color = self.dark_blue
        str_goal = 'In'
        if self.on_button:
            color = self.blue
        if goal == 0:
            str_goal = 'Out'
            color = self.dark_red
            if self.on_button:
                color = self.red
        pygame.draw.rect(self.gameDisplay, color, self.button_goal)
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(str_goal, large_text)
        text_rect.center = (int(self.button_goal[0] + 0.5 * self.button_goal[2]),
                            int(self.button_goal[1] + 0.5 * self.button_goal[3]))
        self.gameDisplay.blit(text_surf, text_rect)

    def game_loop_from_space(self):
        action = -1
        game_exit = False
        done = False
        reward = 0
        self.total = 0
        while not game_exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_ESCAPE:
                        game_exit = True
                        break
                if action != -1:
                    state, reward, done = self.space.action(action)
                    self.total += reward
                if done:
                    self.restart()
                # click = pygame.mouse.get_pressed()
                # mouse = pygame.mouse.get_pos()
                self.on_button = False
                action = -1
            self.gameDisplay.fill(self.black)
            self.draw_space(self.space)
            self.board(self.space.step, reward)
            self.draw_grid()
            self.message_display('Schedule', self.display_width // 2, 80)
            pygame.display.flip()
            self.clock.tick(10)

    def draw_grid(self):
        width = self.width
        height = self.height
        pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init),
                         (self.x_init, self.y_init + self.y_span * height), self.thickness)
        pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init),
                         (self.x_init + self.x_span * width, self.y_init), self.thickness)

        for i in range(width):
            pygame.draw.line(self.gameDisplay, self.blue, (self.x_init + self.x_span * (i + 1), self.y_init),
                             (self.x_init + self.x_span * (i + 1), self.y_init + self.y_span * height), self.thickness)
        for i in range(height):
            pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init + self.y_span * (i + 1)),
                             (self.x_init + self.x_span * width, self.y_init + self.y_span * (i + 1)), self.thickness)

    def draw_space(self, space):
        state = space.get_state()
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:
                    continue
                elif state[i, j] == 1:
                    rgb = self.green
                elif state[i, j] == 2:
                    rgb = self.blue
                self.block(j, i, space.inbound_works[i].id, rgb, x_init=self.x_init)

    def message_display(self, text, x, y):
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(text, large_text)
        text_rect.center = (x, y)
        self.gameDisplay.blit(text_surf, text_rect)


if __name__ == '__main__':
    days = 10
    blocks = 5
    #inbounds = [Work('Work' + str(i), lead_time=-1, max_days=days) for i in range(blocks)]
    inbounds = [Work('Work' + str(i), i // 2, lead_time=-1, max_days=days) for i in range(10)]
    env = Scheduling(num_days=days, num_blocks=blocks, inbound_works=inbounds, display_env=True)
    '''
    s, r, d = env.action(0)
    print(s)
    s, r, d = env.action(1)
    print(s)
    s, r, d = env.action(2)
    print(s)
    s, r, d = env.action(2)
    print(s)
    '''
