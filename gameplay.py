import random
import gamestate, creatures
import math
import pygame
import parl
from ai_agent import AIGameAgent
from ai_model import AIGameModel
from parl.utils import logger, ReplayMemory
import numpy as np


class Gameplay:
    def __init__(self, screen, activated=False):
        self.ourscore = 0
        self.enemyscore = 0
        self.screen = screen
        self.state = gamestate.GameState(self.screen)
        self.activated = activated
        self.ai1 = AIController1()
        # self.ai2 = AIController2()
        # self.ai3 = AIController3()
        # self.ai4 = AIController4()
        # self.ai5 = AIController5()
        # self.ai6 = AIController6()
        self.player = PlayerController()
        self.cempty = EmptyController()
        self.cpreset = PresetController()
        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = [
            self.player, self.cempty, self.cempty, self.ai1, self.cempty, self.cempty]
        self.eventgroup = []
        # self.c1 = self.ai1
        # self.c2 = self.ai2
        # self.c3 = self.ai3
        # self.c4 = self.ai4
        # self.c5 = self.ai5
        # self.c6 = self.ai6

    def setcontroller(self, c1, c2, c3, c4, c5, c6):
        self.c1, self.c2, self.c3, self.c4, self.c5, self.c6 = [c1, c2, c3, c4, c5, c6]

    def activate(self):
        self.activated = True

    def deactivate(self):
        self.activated = False

    def dealevent(self, event):
        if self.activated:
            self.eventgroup.append(event)

    def run(self):
        if self.activated:
            obs1 = self.state.obso1()
            obs2 = self.state.obso2()
            obs3 = self.state.obso3()
            obs4 = self.state.obse1()
            obs5 = self.state.obse2()
            obs6 = self.state.obse3()
            action1 = self.c1.operat(obs1, self.eventgroup)
            action2 = self.c2.operat(obs2, self.eventgroup)
            action3 = self.c3.operat(obs3, self.eventgroup)
            action4 = self.c4.operat(obs4, self.eventgroup)
            action5 = self.c5.operat(obs5, self.eventgroup)
            action6 = self.c6.operat(obs6, self.eventgroup)

            # print('一号已移动至({}, {})'.format(self.state.our1.position[0], self.state.our1.position[1]))
            # print('一号的移动范围限制为{},{}'.format(self.state.our1.moveareax, self.state.our1.moveareay))
            # print('一号的血量为{}, 能量为{}，技能个数为{}'.format(self.state.our1.health, self.state.our1.energy, self.state.our1.skillstorage))
            self.state.our1.update(self.screen,
                                   self.state.sk1group,
                                   self.state.sk2group,
                                   self.state.sk3group,
                                   self.state.sk4group,
                                   action1['chanceofnoskill'],
                                   action1['chanceofskill1'],
                                   action1['chanceofskill2'],
                                   action1['chanceofskill3'],
                                   action1['chanceofskill4'],
                                   action1['skilldirect'],
                                   action1['skillstrength'],
                                   action1['movedirect'],
                                   action1['movestrength'],
                                   )
            self.state.our2.update(self.screen,
                                   self.state.sk1group,
                                   self.state.sk2group,
                                   self.state.sk3group,
                                   self.state.sk4group,
                                   action2['chanceofnoskill'],
                                   action2['chanceofskill1'],
                                   action2['chanceofskill2'],
                                   action2['chanceofskill3'],
                                   action2['chanceofskill4'],
                                   action2['skilldirect'],
                                   action2['skillstrength'],
                                   action2['movedirect'],
                                   action2['movestrength'],
                                   )
            self.state.our3.update(self.screen,
                                   self.state.sk1group,
                                   self.state.sk2group,
                                   self.state.sk3group,
                                   self.state.sk4group,
                                   action3['chanceofnoskill'],
                                   action3['chanceofskill1'],
                                   action3['chanceofskill2'],
                                   action3['chanceofskill3'],
                                   action3['chanceofskill4'],
                                   action3['skilldirect'],
                                   action3['skillstrength'],
                                   action3['movedirect'],
                                   action3['movestrength'],
                                   )
            self.state.enemy1.update(self.screen,
                                     self.state.sk1group,
                                     self.state.sk2group,
                                     self.state.sk3group,
                                     self.state.sk4group,
                                     action4['chanceofnoskill'],
                                     action4['chanceofskill1'],
                                     action4['chanceofskill2'],
                                     action4['chanceofskill3'],
                                     action4['chanceofskill4'],
                                     action4['skilldirect'],
                                     action4['skillstrength'],
                                     action4['movedirect'],
                                     action4['movestrength'],
                                     )
            self.state.enemy2.update(self.screen,
                                     self.state.sk1group,
                                     self.state.sk2group,
                                     self.state.sk3group,
                                     self.state.sk4group,
                                     action5['chanceofnoskill'],
                                     action5['chanceofskill1'],
                                     action5['chanceofskill2'],
                                     action5['chanceofskill3'],
                                     action5['chanceofskill4'],
                                     action5['skilldirect'],
                                     action5['skillstrength'],
                                     action5['movedirect'],
                                     action5['movestrength'],
                                     )
            self.state.enemy3.update(self.screen,
                                     self.state.sk1group,
                                     self.state.sk2group,
                                     self.state.sk3group,
                                     self.state.sk4group,
                                     action6['chanceofnoskill'],
                                     action6['chanceofskill1'],
                                     action6['chanceofskill2'],
                                     action6['chanceofskill3'],
                                     action6['chanceofskill4'],
                                     action6['skilldirect'],
                                     action6['skillstrength'],
                                     action6['movedirect'],
                                     action6['movestrength'],
                                     )
            self.state.sk1group.update(self.screen)
            self.state.sk2group.update(self.screen)
            self.state.sk3group.update(self.screen)
            self.state.sk4group.update(self.screen)
            if not self.state.ourgroup:
                self.enemyscore = 1
            if not self.state.enemygroup:
                self.ourscore = 1


class EmptyController:
    def operat(self, obs, events):
        return {'chanceofnoskill': 1, 'chanceofskill1': 0, 'chanceofskill2': 0,
                'chanceofskill3': 0, 'chanceofskill4': 0,
                'skilldirect': None,
                'skillstrength': None, 'movedirect': None, 'movestrength': None}


class PresetController:
    def operat(self, obs, events):
        nosk, sk1, sk2, sk3, sk4 = 1, 0, 0, 0, 0
        skd = None
        sks = 0
        movd = None
        movs = 0
        rad1 = random.random()
        if obs['myspeedx'] == 0 and obs['myspeedy'] == 0:
            movd = random.random() * 2 * math.pi
        elif obs['myspeedx'] == 0:
            if obs['myspeedy'] > 0:
                movd = 0.5 * math.pi
            else:
                movd = 1.5 * math.pi
        else:
            if obs['myspeedx'] > 0:
                movd = math.atan(obs['myspeedy'] / obs['myspeedx'])
            else:
                movd = math.atan(obs['myspeedy'] / obs['myspeedx']) + math.pi
        if rad1 < 0.8:
            rad2 = 0.4 * (random.random() - 0.5)
            movd = movd + rad2
        else:
            movd = random.random() * 2 * math.pi
        rad = random.random()
        if 0 < obs['ally1health'] < 60 and random.random() > 0.5:
            nosk, sk1, sk2, sk3, sk4, skd, sks = 0, 0, 1, 0, 0, creatures.toangdir(
                obs['myposx'], obs['myposy'], obs['ally1posx'], obs['ally1posy']) / 2 / math.pi, 0.3 * random.random()
        elif 0 < obs['ally2health'] < 60 and random.random() > 0.5:
            nosk, sk1, sk2, sk3, sk4, skd, sks = 0, 0, 1, 0, 0, creatures.toangdir(
                obs['myposx'], obs['myposy'], obs['ally2posx'], obs['ally2posy']) / 2 / math.pi, 0.3 * random.random()
        else:
            if rad < 0.33:
                if not obs['enemy1health'] == 0:
                    nosk, sk1, sk2, sk3, sk4, skd, sks = 0, 1, 0, 0, 0, creatures.toangdir(
                        obs['myposx'], obs['myposy'], obs['enemy1posx'],
                        obs['enemy1posy']) / 2 / math.pi, 0.3 * random.random()
            elif rad > 0.66:
                if not obs['enemy2health'] == 0:
                    nosk, sk1, sk2, sk3, sk4, skd, sks = 0, 1, 0, 0, 0, creatures.toangdir(
                        obs['myposx'], obs['myposy'], obs['enemy2posx'],
                        obs['enemy2posy']) / 2 / math.pi, 0.3 * random.random()
            else:
                if not obs['enemy2health'] == 0:
                    nosk, sk1, sk2, sk3, sk4, skd, sks = 0, 1, 0, 0, 0, creatures.toangdir(
                        obs['myposx'], obs['myposy'], obs['enemy3posx'],
                        obs['enemy3posy']) / 2 / math.pi, 0.3 * random.random()
        # movd = None
        # nosk = 2
        return {'chanceofnoskill': nosk, 'chanceofskill1': sk1, 'chanceofskill2': sk2,
                'chanceofskill3': sk3, 'chanceofskill4': sk4,
                'skilldirect': skd,
                'skillstrength': sks, 'movedirect': movd, 'movestrength': movs}


class PlayerController:
    def __init__(self):
        self.nosk, self.sk1, self.sk2, self.sk3, self.sk4 = 1, 0, 0, 0, 0
        self.skd = None
        self.sks = None
        self.movd = None
        self.movs = None
        self.uping, self.righting, self.downing, self.lefting = False, False, False, False
        self.accing = False
        self.skilltype = 1
        self.skillstrength = 0

    def operat(self, obs, events):
        if events:
            for each in events:
                if each.type == pygame.MOUSEBUTTONDOWN:
                    event = each
                    if event.button == 1:
                        ang = creatures.toangdir(obs['myposx'], obs['myposy'], event.pos[0] - 25, event.pos[1] - 25)
                        direct = ang / 2 / math.pi
                        if self.skilltype == 1:
                            return {'chanceofnoskill': 0, 'chanceofskill1': 1, 'chanceofskill2': 0,
                                    'chanceofskill3': 0, 'chanceofskill4': 0,
                                    'skilldirect': direct,
                                    'skillstrength': self.skillstrength, 'movedirect': None, 'movestrength': None}
                        elif self.skilltype == 2:
                            return {'chanceofnoskill': 0, 'chanceofskill1': 0, 'chanceofskill2': 1,
                                    'chanceofskill3': 0, 'chanceofskill4': 0,
                                    'skilldirect': direct,
                                    'skillstrength': self.skillstrength, 'movedirect': None, 'movestrength': None}
                        elif self.skilltype == 3:
                            return {'chanceofnoskill': 0, 'chanceofskill1': 0, 'chanceofskill2': 0,
                                    'chanceofskill3': 1, 'chanceofskill4': 0,
                                    'skilldirect': direct,
                                    'skillstrength': self.skillstrength, 'movedirect': None, 'movestrength': None}
                        elif self.skilltype == 4:
                            return {'chanceofnoskill': 0, 'chanceofskill1': 0, 'chanceofskill2': 0,
                                    'chanceofskill3': 0, 'chanceofskill4': 1,
                                    'skilldirect': direct,
                                    'skillstrength': self.skillstrength, 'movedirect': None, 'movestrength': None}
                    elif event.button == 3:
                        self.skilltype += 1
                        if self.skilltype > 4:
                            self.skilltype = 1
                    elif event.button == 4:
                        if self.skillstrength < 0.95:
                            self.skillstrength += 0.05
                        else:
                            self.skillstrength = 1
                    elif event.button == 5:
                        if self.skillstrength > 0.05:
                            self.skillstrength -= 0.05
                        else:
                            self.skillstrength = 0
                elif each.type == pygame.KEYDOWN:
                    event = each
                    if event.key == pygame.K_w or event == pygame.K_UP:
                        self.uping = True
                        if event.mod == pygame.KMOD_SHIFT:
                            self.accing = True
                        else:
                            self.accing = False
                    elif event.key == pygame.K_s or event == pygame.K_DOWN:
                        self.downing = True
                        if event.mod == pygame.KMOD_SHIFT:
                            self.accing = True
                        else:
                            self.accing = False
                    elif event.key == pygame.K_a or event == pygame.K_LEFT:
                        self.lefting = True
                        if event.mod == pygame.KMOD_SHIFT:
                            self.accing = True
                        else:
                            self.accing = False
                    elif event.key == pygame.K_d or event == pygame.K_RIGHT:
                        self.righting = True
                        if event.mod == pygame.KMOD_SHIFT:
                            self.accing = True
                        else:
                            self.accing = False
                elif each.type == pygame.KEYUP:
                    event = each
                    if event.key == pygame.K_w or event == pygame.K_UP:
                        self.uping = False
                        self.accing = False
                    elif event.key == pygame.K_s or event == pygame.K_DOWN:
                        self.downing = False
                        self.accing = False
                    elif event.key == pygame.K_a or event == pygame.K_LEFT:
                        self.lefting = False
                        self.accing = False
                    elif event.key == pygame.K_d or event == pygame.K_RIGHT:
                        self.righting = False
                        self.accing = False
                if self.uping:
                    self.movd = 1.5 * math.pi
                    print('在朝上')
                elif self.righting:
                    self.movd = 0
                    print('在朝右')
                elif self.downing:
                    self.movd = 0.5 * math.pi
                    print('在朝下')
                elif self.lefting:
                    self.movd = math.pi
                    print('在朝左')
                elif self.accing:
                    self.movs = 0.
                else:
                    self.movs = 0
                return {'chanceofnoskill': self.nosk, 'chanceofskill1': self.sk1, 'chanceofskill2': self.sk2,
                        'chanceofskill3': self.sk3, 'chanceofskill4': self.sk4,
                        'skilldirect': self.skd,
                        'skillstrength': self.sks, 'movedirect': self.movd, 'movestrength': self.movs}
            else:
                return {'chanceofnoskill': 1, 'chanceofskill1': 0, 'chanceofskill2': 0,
                        'chanceofskill3': 0, 'chanceofskill4': 0,
                        'skilldirect': None,
                        'skillstrength': None, 'movedirect': None, 'movestrength': None}

        else:
            return {'chanceofnoskill': 1, 'chanceofskill1': 0, 'chanceofskill2': 0,
                    'chanceofskill3': 0, 'chanceofskill4': 0,
                    'skilldirect': None,
                    'skillstrength': None, 'movedirect': None, 'movestrength': None}


def tran(s):
    return (s + 1) / 2


class AIController1:
    def __init__(self):
        obs_dim = 62
        act_dim = 9
        ACTOR_LR = 1e-4
        CRITIC_LR = 1e-3
        GAMMA = 0.99
        TAU = 0.001
        model = AIGameModel(act_dim)
        algorithm = parl.algorithms.DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        self.agent = AIGameAgent(algorithm, obs_dim, act_dim)
        self.agent.restore('agent1')

    def operat(self, obs, events):
        obs = np.array(list(obs.values()), dtype='float32')
        action = self.agent.predict(obs.astype('float32'))
        nosk, sk1, sk2, sk3, sk4 = tran(action[0]), tran(action[1]), tran(action[2]), tran(action[3]), tran(action[4])
        skd = tran(action[5])
        sks = tran(action[6])
        movd = tran(action[7])
        movs = tran(action[8])
        return {'chanceofnoskill': nosk, 'chanceofskill1': sk1, 'chanceofskill2': sk2,
                'chanceofskill3': sk3, 'chanceofskill4': sk4,
                'skilldirect': skd,
                'skillstrength': sks, 'movedirect': movd, 'movestrength': movs}


class AIController2:
    def __init__(self):
        obs_dim = 62
        act_dim = 9
        ACTOR_LR = 1e-4
        CRITIC_LR = 1e-3
        GAMMA = 0.99
        TAU = 0.001
        model = AIGameModel(act_dim)
        algorithm = parl.algorithms.DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        self.agent = AIGameAgent(algorithm, obs_dim, act_dim)
        self.agent.restore('agent2')

    def operat(self, obs, events):
        obs = np.array(list(obs.values()), dtype='float32')
        action = self.agent.predict(obs.astype('float32'))
        nosk, sk1, sk2, sk3, sk4 = tran(action[0]), tran(action[1]), tran(action[2]), tran(action[3]), tran(action[4])
        skd = tran(action[5])
        sks = tran(action[6])
        movd = tran(action[7])
        movs = tran(action[8])
        return {'chanceofnoskill': nosk, 'chanceofskill1': sk1, 'chanceofskill2': sk2,
                'chanceofskill3': sk3, 'chanceofskill4': sk4,
                'skilldirect': skd,
                'skillstrength': sks, 'movedirect': movd, 'movestrength': movs}


class AIController3:
    def __init__(self):
        obs_dim = 62
        act_dim = 9
        ACTOR_LR = 1e-4
        CRITIC_LR = 1e-3
        GAMMA = 0.99
        TAU = 0.001
        model = AIGameModel(act_dim)
        algorithm = parl.algorithms.DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        self.agent = AIGameAgent(algorithm, obs_dim, act_dim)
        self.agent.restore('agent3')

    def operat(self, obs, events):
        obs = np.array(list(obs.values()), dtype='float32')
        action = self.agent.predict(obs.astype('float32'))
        nosk, sk1, sk2, sk3, sk4 = tran(action[0]), tran(action[1]), tran(action[2]), tran(action[3]), tran(action[4])
        skd = tran(action[5])
        sks = tran(action[6])
        movd = tran(action[7])
        movs = tran(action[8])
        return {'chanceofnoskill': nosk, 'chanceofskill1': sk1, 'chanceofskill2': sk2,
                'chanceofskill3': sk3, 'chanceofskill4': sk4,
                'skilldirect': skd,
                'skillstrength': sks, 'movedirect': movd, 'movestrength': movs}


class AIController4:
    def __init__(self):
        obs_dim = 62
        act_dim = 9
        ACTOR_LR = 1e-4
        CRITIC_LR = 1e-3
        GAMMA = 0.99
        TAU = 0.001
        model = AIGameModel(act_dim)
        algorithm = parl.algorithms.DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        self.agent = AIGameAgent(algorithm, obs_dim, act_dim)
        self.agent.restore('agent4')

    def operat(self, obs, events):
        obs = np.array(list(obs.values()), dtype='float32')
        action = self.agent.predict(obs.astype('float32'))
        nosk, sk1, sk2, sk3, sk4 = tran(action[0]), tran(action[1]), tran(action[2]), tran(action[3]), tran(action[4])
        skd = tran(action[5])
        sks = tran(action[6])
        movd = tran(action[7])
        movs = tran(action[8])
        return {'chanceofnoskill': nosk, 'chanceofskill1': sk1, 'chanceofskill2': sk2,
                'chanceofskill3': sk3, 'chanceofskill4': sk4,
                'skilldirect': skd,
                'skillstrength': sks, 'movedirect': movd, 'movestrength': movs}


class AIController5:
    def __init__(self):
        obs_dim = 62
        act_dim = 9
        ACTOR_LR = 1e-4
        CRITIC_LR = 1e-3
        GAMMA = 0.99
        TAU = 0.001
        model = AIGameModel(act_dim)
        algorithm = parl.algorithms.DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        self.agent = AIGameAgent(algorithm, obs_dim, act_dim)
        self.agent.restore('agent5')

    def operat(self, obs, events):
        obs = np.array(list(obs.values()), dtype='float32')
        action = self.agent.predict(obs.astype('float32'))
        nosk, sk1, sk2, sk3, sk4 = tran(action[0]), tran(action[1]), tran(action[2]), tran(action[3]), tran(action[4])
        skd = tran(action[5])
        sks = tran(action[6])
        movd = tran(action[7])
        movs = tran(action[8])
        return {'chanceofnoskill': nosk, 'chanceofskill1': sk1, 'chanceofskill2': sk2,
                'chanceofskill3': sk3, 'chanceofskill4': sk4,
                'skilldirect': skd,
                'skillstrength': sks, 'movedirect': movd, 'movestrength': movs}


class AIController6:
    def __init__(self):
        obs_dim = 62
        act_dim = 9
        ACTOR_LR = 1e-4
        CRITIC_LR = 1e-3
        GAMMA = 0.99
        TAU = 0.001
        model = AIGameModel(act_dim)
        algorithm = parl.algorithms.DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        self.agent = AIGameAgent(algorithm, obs_dim, act_dim)
        self.agent.restore('agent6')

    def operat(self, obs, events):
        obs = np.array(list(obs.values()), dtype='float32')
        action = self.agent.predict(obs.astype('float32'))
        nosk, sk1, sk2, sk3, sk4 = tran(action[0]), tran(action[1]), tran(action[2]), tran(action[3]), tran(action[4])
        skd = tran(action[5])
        sks = tran(action[6])
        movd = tran(action[7])
        movs = tran(action[8])
        return {'chanceofnoskill': nosk, 'chanceofskill1': sk1, 'chanceofskill2': sk2,
                'chanceofskill3': sk3, 'chanceofskill4': sk4,
                'skilldirect': skd,
                'skillstrength': sks, 'movedirect': movd, 'movestrength': movs}
