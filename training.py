import gamestate
import parl
from ai_agent import AIGameAgent
from ai_model import AIGameModel
from parl.utils import logger, ReplayMemory
import numpy as np
import pygame
import gamestate
import math, random
import creatures

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.001
MEMORY_SIZE = int(1e6)
MEMORY_WARMUP_SIZE = 2000
BATCH_SIZE = 128
REWARD_SCALE = 0.1
ENV_SEED = 1


def tran(x):
    return (x + 1) / 2


def run_train_episode(agent1, rpm, screen, state):
    state.reset()
    obso1 = np.array(list(state.obso1().values()), dtype='float32')
    total_rewardo1 = 0
    times = 0
    while True:
        batch_obso1 = np.expand_dims(obso1, axis=0)
        actiono1 = agent1.predict(batch_obso1.astype('float32'))

        # Add exploration noise, and clip to [-1.0, 1.0]
        actiono1 = np.clip(np.random.normal(actiono1, 0.3), -1.0, 1.0)
        actione1 = defaultoperat(state.obse1())

        # next_obs, reward, done, info = env.step(action)
        state.our1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono1[0]),
                          tran(actiono1[1]), tran(actiono1[2]), tran(actiono1[3]), tran(actiono1[4]),
                          tran(actiono1[5]), tran(actiono1[6]), tran(actiono1[7]), tran(actiono1[8]))
        state.our2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                          1, 0, 0, 0, 0, None, None, None, None)
        state.our3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                          1, 0, 0, 0, 0, None, None, None, None)
        state.enemy1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                            actione1['chanceofnoskill'],
                            actione1['chanceofskill1'],
                            actione1['chanceofskill2'],
                            actione1['chanceofskill3'],
                            actione1['chanceofskill4'],
                            actione1['skilldirect'],
                            actione1['skillstrength'],
                            actione1['movedirect'],
                            actione1['movestrength'], )
        # state.enemy1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
        #                     1, 0, 0, 0, 0, None, None, None, None)
        state.enemy2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                            1, 0, 0, 0, 0, None, None, None, None)
        state.enemy3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                            1, 0, 0, 0, 0, None, None, None, None)

        state.sk1group.update(screen)
        state.sk2group.update(screen)
        state.sk3group.update(screen)
        state.sk4group.update(screen)

        next_obso1 = np.array(list(state.obso1().values()), dtype='float32')

        # if state.our1.health == 0:
        #     reward_o1 = -1000
        #     done = True
        # elif state.our2.health == 0:
        #     reward_o1 = -200
        #     done = True
        # elif state.our3.health == 0:
        #     reward_o1 = -200
        #     done = True
        # elif state.enemy1.health == 0:
        #     reward_o1 = 1000
        #     done = True
        # elif state.enemy2.health == 0:
        #     reward_o1 = 1000
        #     done = True
        # elif state.enemy3.health == 0:
        #     reward_o1 = 1000
        #     done = True
        # elif times >= 1000:
        #     reward_o1 = -1
        #     done = True
        # else:
        #     reward_o1 = -1
        #     done = False

        done = False
        if state.our1.health <= 0:
            done = True
        elif state.enemy1.health <= 0 and state.enemy2.health <= 0 and state.enemy3.health <= 0:
            reward_o1 = 10000000
            done = True
        elif times >= 1000:
            done = True
        reward_o1 = (state.our1.health + (state.our2.health + state.our3.health) / 2 - 10 * (
                state.enemy1.health + state.enemy2.health + state.enemy3.health) + 4000) / 1000

        rpm.append(obso1, actiono1, REWARD_SCALE * reward_o1, next_obso1, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            # batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            #     BATCH_SIZE)
            # agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
            #             batch_terminal)
            batch_obso1, batch_actiono1, batch_rewardo1, batch_next_obso1, batch_terminalo1 = rpm.sample_batch(
                BATCH_SIZE)
            agent1.learn(batch_obso1, batch_actiono1, batch_rewardo1, batch_next_obso1, batch_terminalo1)

        obso1 = next_obso1
        # total_reward += reward
        total_rewardo1 += reward_o1
        times += 1
        if done:
            print('完成了一场游戏')
            break
    return total_rewardo1


def run_evaluate_episode(agent1, screen, state):
    state.reset()
    obso1 = np.array(list(state.obso1().values()), dtype='float32')
    total_rewardo1 = 0
    times = 0
    while True:
        batch_obso1 = np.expand_dims(obso1, axis=0)
        actiono1 = agent1.predict(batch_obso1.astype('float32'))

        # next_obs, reward, done, info = env.step(action)
        state.our1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono1[0]),
                          tran(actiono1[1]), tran(actiono1[2]), tran(actiono1[3]), tran(actiono1[4]),
                          tran(actiono1[5]), tran(actiono1[6]), tran(actiono1[7]), tran(actiono1[8]))
        state.our2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                          1, 0, 0, 0, 0, None, None, None, None)
        state.our3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                          1, 0, 0, 0, 0, None, None, None, None)
        state.enemy1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                            1, 0, 0, 0, 0, None, None, None, None)
        state.enemy2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                            1, 0, 0, 0, 0, None, None, None, None)
        state.enemy3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group,
                            1, 0, 0, 0, 0, None, None, None, None)
        state.sk1group.update(screen)
        state.sk2group.update(screen)
        state.sk3group.update(screen)
        state.sk4group.update(screen)
        next_obso1 = np.array(list(state.obso1().values()), dtype='float32')

        if state.our1.health == 0:
            reward_o1 = -1000
            done = True
        elif state.our2.health == 0:
            reward_o1 = -200
            done = True
        elif state.our3.health == 0:
            reward_o1 = -200
            done = True
        elif state.enemy1.health == 0:
            reward_o1 = 1000
            done = True
        elif state.enemy2.health == 0:
            reward_o1 = 1000
            done = True
        elif state.enemy3.health == 0:
            reward_o1 = 1000
            done = True
        elif times >= 1000:
            reward_o1 = -1
            done = True
        else:
            reward_o1 = -1
            done = False

        obso1 = next_obso1
        # total_reward += reward
        total_rewardo1 += reward_o1

        if done:
            break
    return total_rewardo1


def totrain(screen):
    train_total_episode = 1e7
    obs_dim = 62
    act_dim = 9
    state = gamestate.GameState(screen)
    model1 = AIGameModel(act_dim)
    algorithm1 = parl.algorithms.DDPG(
        model1, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent1 = AIGameAgent(algorithm1, obs_dim, act_dim)
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)
    agent1.restore('agent1')
    while rpm.size() < MEMORY_WARMUP_SIZE:
        run_train_episode(agent1, rpm, screen, state)

    episode = 0
    while episode < train_total_episode:
        for i in range(10):
            train_rewardo1 = run_train_episode(agent1, rpm, screen, state)
            episode += 1
            logger.info('Episode: {} Reward1: {}'.format(episode, train_rewardo1))
        agent1.save('agent1')
        print('已保存训练数据')
        # evaluate_rewardo1 = run_evaluate_episode(agent1, screen, state)
        # logger.info('Episode {}, Evaluate reward1: {}'.format(episode, evaluate_rewardo1))


def defaultoperat(obs):
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
