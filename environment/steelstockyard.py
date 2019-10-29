from random import randint
import pygame
import time


# 강재 정보 클래스 id, 입출고일정 포함
class Plate(object):
    def __init__(self, plate_id=None, inbound=0, outbound=1):
        self.id = str(plate_id)
        self.inbound = inbound
        if outbound == -1:  # 강재 데이터가 없으면 임의로 출고일 생성
            self.outbound = randint(0, 5)


# 강재 적치 위치 결정 환경
class Locating(object):  # 생성자에서 파일의 수, 최대 높이 등을 입력
    def __init__(self, num_pile=4, max_stack=4, inbound_plates=None, display_env=False):
        self.action_space = num_pile  # 가능한 action 수는 파일의 수로 설정
        self.max_stack = max_stack  # 한 파일에 적치 가능한 강재의 수
        self.empty = 0  # 빈 공간의 상태 표현 값
        self.step = 0
        self.inbound_plates = inbound_plates
        self.inbound_clone = inbound_plates[:]
        self.plates = [[] for _ in range(num_pile)]  # 각 파일을 빈 리스트로 초기화
        # self.yard = np.full([max_stack, num_pile], self.empty)
        if display_env:  # 환경을 게임엔진으로 가시화하는 용도. 학습용시에는 사용하지 않음
            display = LocatingDisplay(self, num_pile, max_stack, 2)
            display.game_loop_from_space()

    def action(self, action):
        done = False
        reward = 0
        inbound = self.inbound_plates.pop()
        self.plates[action].append(inbound)
        reward = self._calculate_reward(action)
        self.step += 1
        if len(self.inbound_plates) == 0:
            done = True
        next_state = []
        return next_state, reward, done

    def reset(self):
        self.inbound_plates = self.inbound_clone[:]
        self.plates = [[] for _ in range(self.action_space)]
        self.step = 0

    def _calculate_reward(self, action):
        pile = self.plates[action]
        max_move = 0
        if len(pile) == 1:
            return 0
        for i, plate in enumerate(pile[:-1]):
            move = 0
            if i + max_move > len(pile):
                break
            for upper in pile[i + 1:]:
                if plate.outbound < upper.outbound:
                    move += 1
            if move > max_move:
                max_move = move
        reward = 2
        if max_move != 0:
            reward = 1 / max_move
        return reward



# 환경을 가시화하는 용도, 사람이 action 을 입력해야하므로 학습시에는 실행하지 않음
class LocatingDisplay(object):
    white = (255, 255, 255)
    black = (0, 0, 0)

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    dark_red = (200, 0, 0)
    dark_green = (0, 200, 0)
    dark_blue = (0, 0, 200)

    x_init = 300
    y_init = 100
    x_span = 100
    y_span = 30
    thickness = 5
    pygame.init()
    display_width = 1000
    display_height = 600
    font = 'freesansbold.ttf'
    pygame.display.set_caption('Steel Locating')
    clock = pygame.time.Clock()
    pygame.key.set_repeat()
    button_goal = (display_width - 100, 10, 70, 40)

    def __init__(self, locating, width, height, num_block):
        self.width = width
        self.height = height
        self.num_block = num_block
        self.space = locating
        self.on_button = False
        self.total = 0
        self.display_width = 150 * width + 200
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
        text_surf, text_rect = self.text_objects('step: ' + str(step) + '   reward: ' + format(reward, '.2f')
                                                 + '   total: ' + format(self.total, '.2f'), large_text)
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
                    if event.key == pygame.K_1:
                        action = 0
                    elif event.key == pygame.K_2:
                        action = 1
                    elif event.key == pygame.K_3:
                        action = 2
                    elif event.key == pygame.K_4:
                        action = 3
                    elif event.key == pygame.K_5:
                        action = 4
                    elif event.key == pygame.K_6:
                        action = 5
                    elif event.key == pygame.K_7:
                        action = 6
                    elif event.key == pygame.K_8:
                        action = 7
                    elif event.key == pygame.K_ESCAPE:
                        game_exit = True
                        break
                if action != -1:
                    _, reward, done = self.space.action(action)
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
            self.draw_grid(self.width, 1, self.x_init, self.y_init, self.x_span, self.y_span * self.height)
            self.draw_grid(1, 1, 100, 100, self.x_span, self.y_span * 10)
            self.message_display('Inbound plates', 150, 80)
            self.message_display('Stockyard', 500, 80)
            pygame.display.flip()
            self.clock.tick(10)

    def draw_grid(self, width, height, x_init, y_init, x_span, y_span):
        pygame.draw.line(self.gameDisplay, self.blue, (x_init, y_init),
                         (x_init, y_init + y_span * height), self.thickness)
        pygame.draw.line(self.gameDisplay, self.blue, (x_init, y_init),
                         (x_init + x_span * width, y_init), self.thickness)
        for i in range(width):
            pygame.draw.line(self.gameDisplay, self.blue, (x_init + x_span * (i + 1), y_init),
                             (x_init + x_span * (i + 1), y_init + y_span * height), self.thickness)
        for i in range(height):
            pygame.draw.line(self.gameDisplay, self.blue, (x_init, y_init + y_span * (i + 1)),
                             (x_init + x_span * width, y_init + y_span * (i + 1)), self.thickness)

    def draw_space(self, space):
        for i, pile in enumerate(space.plates):
            for j, plate in enumerate(pile):
                rgb = 150 * (1 / max(1, plate.outbound - plate.inbound))
                self.block(i, self.space.max_stack - j - 1, plate.id, (rgb, rgb, rgb), x_init=self.x_init)
        for i, plate in enumerate(space.inbound_plates[-10:]):
            rgb = 150 * (1 / max(1, plate.outbound - plate.inbound))
            self.block(0, self.space.max_stack - i - 1, plate.id, (rgb, rgb, rgb), x_init=100)

    def message_display(self, text, x, y):
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(text, large_text)
        text_rect.center = (x, y)
        self.gameDisplay.blit(text_surf, text_rect)


# 환경 가시화 및 테스트시에 사용하는 코드
if __name__ == '__main__':
    inbounds = [Plate('P' + str(i), outbound=-1) for i in range(30)]  # 테스트용 임의 강재 데이터
    s = Locating(max_stack=10, num_pile=8, inbound_plates=inbounds, display_env=True)  # 환경 테스트
    print(s.plates)
