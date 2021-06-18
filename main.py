import pygame, sys, traceback
import gui
import gameplay
import ai_train
import training
from pygame import *

pygame.init()
pygame.mixer.init()

size = width, height = 960, 540
screen = pygame.display.set_mode(size)
pygame.display.set_caption("AI训练游戏(暂定) Demo")
fps = 30


def main():
    clock = pygame.time.Clock()
    running = True
    game = gameplay.Gameplay(screen)
    buttongroup1 = gui.ButtonGroup((200, 200, 200), screen, True,
                                   gui.Button(text='开始训练'),
                                   gui.Button(text='开始挑战'),
                                   gui.Button(text='设置天赋'),
                                   gui.Button(text='亲自上阵'))
    buttongroup2 = gui.ButtonGroup((200, 200, 200), screen, False,
                                   gui.Button(text='与敌人训练'),
                                   gui.Button(text='与自己训练'),
                                   gui.Button(text='与预设敌人训练'),
                                   gui.Button(text='返回'))
    buttongroup3 = gui.ButtonGroup((200, 200, 200), screen, False,
                                   gui.Button(text='挑战对手'),
                                   gui.Button(text='挑战自己'),
                                   gui.Button(text='挑战预设敌人'),
                                   gui.Button(text='返回'))
    buttongroup4 = gui.ButtonGroup((200, 200, 200), screen, False,
                                   gui.Button(text='扮演一号'),
                                   gui.Button(text='扮演二号'),
                                   gui.Button(text='扮演三号'),
                                   gui.Button(text='返回'))
    buttongroup5 = gui.ButtonGroup((200, 200, 200), screen, False,
                                   gui.Button(text='亲自挑战自己的队伍'),
                                   gui.Button(text='亲自挑战对手'),
                                   gui.Button(text='亲自挑战预设敌人'),
                                   gui.Button(text='返回'))

    while running:
        for event in pygame.event.get():
            game.dealevent(event)
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                buttongroup1.clicked(event)
                buttongroup2.clicked(event)
                buttongroup3.clicked(event)
                buttongroup4.clicked(event)
                buttongroup5.clicked(event)
            if event.type == pygame.USEREVENT + 1:
                if event.text == '开始训练':
                    buttongroup1.deactivate()
                    buttongroup2.activate()
                elif event.text == '开始挑战':
                    buttongroup1.deactivate()
                    buttongroup3.activate()
                elif event.text == '亲自上阵':
                    buttongroup1.deactivate()
                    buttongroup4.activate()
                elif event.text == '扮演一号':
                    buttongroup4.deactivate()
                    buttongroup5.activate()
                    playerNum = 1
                elif event.text == '扮演二号':
                    buttongroup4.deactivate()
                    buttongroup5.activate()
                    playerNum = 2
                elif event.text == '扮演三号':
                    buttongroup4.deactivate()
                    buttongroup5.activate()
                    playerNum = 3
                elif event.text == '亲自挑战预设敌人':
                    buttongroup5.deactivate()
                    if playerNum == 1:
                        print('已启动游戏')
                        game.activate()
                    elif playerNum == 2:
                        buttongroup5.deactivate()
                        training.totrain(screen)
                    elif playerNum == 3:
                        pass
                elif event.text == '亲自挑战对手':
                    buttongroup5.deactivate()
                    if playerNum == 1:
                        pass
                    elif playerNum == 2:
                        pass
                    elif playerNum == 3:
                        pass
                elif event.text == '亲自挑战自己的队伍':
                    buttongroup5.deactivate()
                    if playerNum == 1:
                        pass
                    elif playerNum == 2:
                        pass
                    elif playerNum == 3:
                        pass
                elif event.text == '返回':
                    if buttongroup2.isActive:
                        buttongroup2.deactivate()
                        buttongroup1.activate()
                    elif buttongroup3.isActive:
                        buttongroup3.deactivate()
                        buttongroup1.activate()
                    elif buttongroup4.isActive:
                        buttongroup4.deactivate()
                        buttongroup1.activate()
                    elif buttongroup5.isActive:
                        buttongroup5.deactivate()
                        buttongroup4.activate()

        screen.fill('white')
        game.run()
        game.eventgroup.clear()
        buttongroup1.draw()
        buttongroup2.draw()
        buttongroup3.draw()
        buttongroup4.draw()
        buttongroup5.draw()
        pygame.display.flip()
        clock.tick(fps)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
        input()
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
