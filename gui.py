import pygame
import pygame.freetype


class Button:
    def __init__(self, text, color='red', area=[100, 20], pos=[0, 0]):
        self.pos = pos
        self.area = area
        self.color = color
        self.isActive = True
        self.surface = pygame.Surface(area)
        self.rect = pygame.Rect(pos, area)
        self.font = pygame.freetype.Font(r"C:\Windows\Fonts\simsun.ttc", 25)
        self.text = text

    def action(self):
        pygame.event.post(pygame.event.Event(pygame.USEREVENT + 1, {'text': self.text}))

    def getArea(self):
        return self.pos, self.area

    def draw(self, screen):
        self.surface.fill((200, 200, 200))
        self.font.render_to(self.surface, [10, 5], self.text, fgcolor=self.color, bgcolor=(200, 200, 200))
        screen.blit(self.surface, self.rect)

    def deactivate(self):
        self.isActive = False

    def activate(self):
        self.isActive = True


class ButtonGroup:
    def __init__(self, color, screen, active=True, *buttons):
        self.color = color
        self.buttons = buttons
        self.screen = screen
        self.isActive = active
        self.butSize = [1000, 30]
        self.butGap = 40
        self.font = pygame.freetype.Font(r"C:\Windows\Fonts\simsun.ttc", 25)

    def clicked(self, event):
        if self.isActive:
            x, y = event.pos
            butPos = [20, 10]
            for button in self.buttons:
                rect = pygame.Rect(butPos, self.butSize)
                if rect.collidepoint(x, y):
                    button.action()
                    # print('按钮%s区域被点击' % button.text)
                    self.draw()
                butPos[1] += self.butGap

    def draw(self):
        if self.isActive:
            self.screen.fill('white')
            butPos = [20, 10]
            for button in self.buttons:
                surface = pygame.Surface(self.butSize)
                rect = pygame.Rect(butPos, self.butSize)
                surface.fill(self.color)
                self.font.render_to(surface, [10, 5], button.text, fgcolor=button.color, bgcolor=self.color)
                self.screen.blit(surface, rect)
                butPos[1] += self.butGap

    def deactivate(self):
        self.isActive = False

    def activate(self):
        self.isActive = True
