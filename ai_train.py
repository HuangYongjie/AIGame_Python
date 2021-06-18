import gamestate
import parl
from ai_agent import AIGameAgent
from ai_model import AIGameModel
from parl.utils import logger, ReplayMemory
import numpy as np
import pygame
import gamestate

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.001
MEMORY_SIZE = int(1e5)
MEMORY_WARMUP_SIZE = 500
BATCH_SIZE = 128
REWARD_SCALE = 0.1
ENV_SEED = 1


def tran(arg):
    return (arg + 1) / 2


def run_train_episode(agent1, agent2, agent3, agent4, agent5, agent6, rpm, screen, state):
    # obs = env.reset()
    times = 0
    state.reset()
    obso1 = np.array(list(state.obso1().values()), dtype='float32')
    obso2 = np.array(list(state.obso2().values()), dtype='float32')
    obso3 = np.array(list(state.obso3().values()), dtype='float32')
    obse1 = np.array(list(state.obse1().values()), dtype='float32')
    obse2 = np.array(list(state.obse2().values()), dtype='float32')
    obse3 = np.array(list(state.obse3().values()), dtype='float32')
    total_rewardo1 = 0
    total_rewardo2 = 0
    total_rewardo3 = 0
    total_rewarde1 = 0
    total_rewarde2 = 0
    total_rewarde3 = 0
    while True:
        # batch_obs = np.expand_dims(obs, axis=0)
        batch_obso1 = np.expand_dims(obso1, axis=0)
        batch_obso2 = np.expand_dims(obso2, axis=0)
        batch_obso3 = np.expand_dims(obso3, axis=0)
        batch_obse1 = np.expand_dims(obse1, axis=0)
        batch_obse2 = np.expand_dims(obse2, axis=0)
        batch_obse3 = np.expand_dims(obse3, axis=0)
        actiono1 = agent1.predict(batch_obso1.astype('float32'))
        actiono2 = agent2.predict(batch_obso2.astype('float32'))
        actiono3 = agent3.predict(batch_obso3.astype('float32'))
        actione1 = agent4.predict(batch_obse1.astype('float32'))
        actione2 = agent5.predict(batch_obse2.astype('float32'))
        actione3 = agent6.predict(batch_obse3.astype('float32'))

        # Add exploration noise, and clip to [-1.0, 1.0]
        # action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        actiono1 = np.clip(np.random.normal(actiono1, 0.6), -1.0, 1.0)
        actiono2 = np.clip(np.random.normal(actiono2, 0.6), -1.0, 1.0)
        actiono3 = np.clip(np.random.normal(actiono3, 0.6), -1.0, 1.0)
        actione1 = np.clip(np.random.normal(actione1, 0.6), -1.0, 1.0)
        actione2 = np.clip(np.random.normal(actione2, 0.6), -1.0, 1.0)
        actione3 = np.clip(np.random.normal(actione3, 0.6), -1.0, 1.0)

        # next_obs, reward, done, info = env.step(action)
        state.our1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono1[0]),
                          tran(actiono1[1]), tran(actiono1[2]), tran(actiono1[3]), tran(actiono1[4]),
                          tran(actiono1[5]), tran(actiono1[6]), tran(actiono1[7]), tran(actiono1[8]))
        state.our2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono2[0]),
                          tran(actiono2[1]), tran(actiono2[2]), tran(actiono2[3]), tran(actiono2[4]),
                          tran(actiono2[5]), tran(actiono2[6]), tran(actiono2[7]), tran(actiono2[8]))
        state.our3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono3[0]),
                          tran(actiono3[1]), tran(actiono3[2]), tran(actiono3[3]), tran(actiono3[4]),
                          tran(actiono3[5]), tran(actiono3[6]), tran(actiono3[7]), tran(actiono3[8]))
        state.enemy1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actione1[0]),
                          tran(actione1[1]), tran(actione1[2]), tran(actione1[3]), tran(actione1[4]),
                          tran(actione1[5]), tran(actione1[6]), tran(actione1[7]), tran(actione1[8]))
        state.enemy2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actione2[0]),
                          tran(actione2[1]), tran(actione2[2]), tran(actione2[3]), tran(actione2[4]),
                          tran(actione2[5]), tran(actione2[6]), tran(actione2[7]), tran(actione2[8]))
        state.enemy3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actione3[0]),
                          tran(actione3[1]), tran(actione3[2]), tran(actione3[3]), tran(actione3[4]),
                          tran(actione3[5]), tran(actione3[6]), tran(actione3[7]), tran(actione3[8]))

        state.sk1group.update(screen)
        state.sk2group.update(screen)
        state.sk3group.update(screen)
        state.sk4group.update(screen)

        next_obso1 = np.array(list(state.obso1().values()), dtype='float32')
        next_obso2 = np.array(list(state.obso2().values()), dtype='float32')
        next_obso3 = np.array(list(state.obso3().values()), dtype='float32')
        next_obse1 = np.array(list(state.obse1().values()), dtype='float32')
        next_obse2 = np.array(list(state.obse2().values()), dtype='float32')
        next_obse3 = np.array(list(state.obse3().values()), dtype='float32')
        #
        # if not state.enemygroup and state.ourgroup:
        #     reward_o1, reward_o2, reward_o3 = 1, 1, 1
        #     reward_e1, reward_e2, reward_e3 = -0.5, -0.5, -0.5
        #     done = True
        # elif not state.ourgroup and state.enemygroup:
        #     reward_e1, reward_e2, reward_e3 = 1, 1, 1
        #     reward_o1, reward_o2, reward_o3 = -0.5, -0.5, -0.5
        #     done = True
        # elif state.ourgroup and state.enemygroup:
        #     # print('我方还有{}名'.format(len(state.ourgroup)))
        #     # print('敌方还有{}名'.format(len(state.enemygroup)))
        #     reward_o1, reward_o2, reward_o3 = 0, 0, 0
        #     reward_e1, reward_e2, reward_e3 = 0, 0, 0
        #     done = False
        # else:
        #     reward_o1, reward_o2, reward_o3 = 0, 0, 0
        #     reward_e1, reward_e2, reward_e3 = 0, 0, 0
        #     done = True
        reward_o1, reward_o2, reward_o3, reward_e1, reward_e2, reward_e3 = 0, 0, 0, 0, 0, 0
        if state.our1.health == 0:
            reward_o1 = -1
            reward_e1 = 0.2
            reward_e2 = 0.2
            reward_e3 = 0.2
            done = True
        elif state.our2.health == 0:
            reward_o2 = -1
            reward_e1 = 0.2
            reward_e2 = 0.2
            reward_e3 = 0.2
            done = True
        elif state.our3.health == 0:
            reward_o3 = -1
            reward_e1 = 0.2
            reward_e2 = 0.2
            reward_e3 = 0.2
            done = True
        elif state.enemy1.health == 0:
            reward_e1 = -1
            reward_o1 = 0.2
            reward_o2 = 0.2
            reward_o3 = 0.2
            done = True
        elif state.enemy2.health == 0:
            reward_e2 = -1
            reward_o1 = 0.2
            reward_o2 = 0.2
            reward_o3 = 0.2
            done = True
        elif state.enemy3.health == 0:
            reward_e3 = -1
            reward_o1 = 0.2
            reward_o2 = 0.2
            reward_o3 = 0.2
            done = True
        elif times >= 2000:
            done = True
        else:
            done = False
        times += 1
        rpm.append(obso1, actiono1, REWARD_SCALE * reward_o1, next_obso1,
                   obso2, actiono2, REWARD_SCALE * reward_o2, next_obso2,
                   obso3, actiono3, REWARD_SCALE * reward_o3, next_obso3,
                   obse1, actione1, REWARD_SCALE * reward_e1, next_obse1,
                   obse2, actione2, REWARD_SCALE * reward_e2, next_obse2,
                   obse3, actione3, REWARD_SCALE * reward_e3, next_obse3,
                   done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            # batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            #     BATCH_SIZE)
            # agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
            #             batch_terminal)
            batch_obso1, batch_actiono1, batch_rewardo1, batch_next_obso1, batch_terminalo1 = rpm.sample_batch_o1(
                BATCH_SIZE)
            agent1.learn(batch_obso1, batch_actiono1, batch_rewardo1, batch_next_obso1, batch_terminalo1)
            batch_obso2, batch_actiono2, batch_rewardo2, batch_next_obso2, batch_terminalo2 = rpm.sample_batch_o2(
                BATCH_SIZE)
            agent2.learn(batch_obso2, batch_actiono2, batch_rewardo2, batch_next_obso2, batch_terminalo2)
            batch_obso3, batch_actiono3, batch_rewardo3, batch_next_obso3, batch_terminalo3 = rpm.sample_batch_o3(
                BATCH_SIZE)
            agent3.learn(batch_obso3, batch_actiono3, batch_rewardo3, batch_next_obso3, batch_terminalo3)
            batch_obse1, batch_actione1, batch_rewarde1, batch_next_obse1, batch_terminale1 = rpm.sample_batch_e1(
                BATCH_SIZE)
            agent4.learn(batch_obse1, batch_actione1, batch_rewarde1, batch_next_obse1, batch_terminale1)
            batch_obse2, batch_actione2, batch_rewarde2, batch_next_obse2, batch_terminale2 = rpm.sample_batch_e2(
                BATCH_SIZE)
            agent5.learn(batch_obse2, batch_actione2, batch_rewarde2, batch_next_obse2, batch_terminale2)
            batch_obse3, batch_actione3, batch_rewarde3, batch_next_obse3, batch_terminale3 = rpm.sample_batch_e3(
                BATCH_SIZE)
            agent6.learn(batch_obse3, batch_actione3, batch_rewarde3, batch_next_obse3, batch_terminale3)

        obso1 = next_obso1
        obso2 = next_obso2
        obso3 = next_obso3
        obse1 = next_obse1
        obse2 = next_obse2
        obse3 = next_obse3
        # total_reward += reward
        total_rewardo1 += reward_o1
        total_rewardo2 += reward_o2
        total_rewardo3 += reward_o3
        total_rewarde1 += reward_e1
        total_rewarde2 += reward_e2
        total_rewarde3 += reward_e3

        if done:
            break
    return total_rewardo1, total_rewardo2, total_rewardo3, total_rewarde1, total_rewarde2, total_rewarde3


def run_evaluate_episode(agent1, agent2, agent3, agent4, agent5, agent6, rpm, screen, state):
    # obs = env.reset()

    times = 0
    state.reset()
    obso1 = np.array(list(state.obso1().values()), dtype='float32')
    obso2 = np.array(list(state.obso2().values()), dtype='float32')
    obso3 = np.array(list(state.obso3().values()), dtype='float32')
    obse1 = np.array(list(state.obse1().values()), dtype='float32')
    obse2 = np.array(list(state.obse2().values()), dtype='float32')
    obse3 = np.array(list(state.obse3().values()), dtype='float32')
    total_rewardo1 = 0
    total_rewardo2 = 0
    total_rewardo3 = 0
    total_rewarde1 = 0
    total_rewarde2 = 0
    total_rewarde3 = 0
    while True:
        # batch_obs = np.expand_dims(obs, axis=0)
        batch_obso1 = np.expand_dims(obso1, axis=0)
        batch_obso2 = np.expand_dims(obso2, axis=0)
        batch_obso3 = np.expand_dims(obso3, axis=0)
        batch_obse1 = np.expand_dims(obse1, axis=0)
        batch_obse2 = np.expand_dims(obse2, axis=0)
        batch_obse3 = np.expand_dims(obse3, axis=0)
        actiono1 = agent1.predict(batch_obso1.astype('float32'))
        actiono2 = agent2.predict(batch_obso2.astype('float32'))
        actiono3 = agent3.predict(batch_obso3.astype('float32'))
        actione1 = agent4.predict(batch_obse1.astype('float32'))
        actione2 = agent5.predict(batch_obse2.astype('float32'))
        actione3 = agent6.predict(batch_obse3.astype('float32'))
        # next_obs, reward, done, info = env.step(action)
        # next_obs, reward, done, info = env.step(action)
        state.our1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono1[0]),
                          tran(actiono1[1]), tran(actiono1[2]), tran(actiono1[3]), tran(actiono1[4]),
                          tran(actiono1[5]), tran(actiono1[6]), tran(actiono1[7]), tran(actiono1[8]))
        state.our2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono2[0]),
                          tran(actiono2[1]), tran(actiono2[2]), tran(actiono2[3]), tran(actiono2[4]),
                          tran(actiono2[5]), tran(actiono2[6]), tran(actiono2[7]), tran(actiono2[8]))
        state.our3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actiono3[0]),
                          tran(actiono3[1]), tran(actiono3[2]), tran(actiono3[3]), tran(actiono3[4]),
                          tran(actiono3[5]), tran(actiono3[6]), tran(actiono3[7]), tran(actiono3[8]))
        state.enemy1.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actione1[0]),
                          tran(actione1[1]), tran(actione1[2]), tran(actione1[3]), tran(actione1[4]),
                          tran(actione1[5]), tran(actione1[6]), tran(actione1[7]), tran(actione1[8]))
        state.enemy2.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actione2[0]),
                          tran(actione2[1]), tran(actione2[2]), tran(actione2[3]), tran(actione2[4]),
                          tran(actione2[5]), tran(actione2[6]), tran(actione2[7]), tran(actione2[8]))
        state.enemy3.update(screen, state.sk1group, state.sk2group, state.sk3group, state.sk4group, tran(actione3[0]),
                          tran(actione3[1]), tran(actione3[2]), tran(actione3[3]), tran(actione3[4]),
                          tran(actione3[5]), tran(actione3[6]), tran(actione3[7]), tran(actione3[8]))

        state.sk1group.update(screen)
        state.sk2group.update(screen)
        state.sk3group.update(screen)
        state.sk4group.update(screen)

        next_obso1 = np.array(list(state.obso1().values()), dtype='float32')
        next_obso2 = np.array(list(state.obso2().values()), dtype='float32')
        next_obso3 = np.array(list(state.obso3().values()), dtype='float32')
        next_obse1 = np.array(list(state.obse1().values()), dtype='float32')
        next_obse2 = np.array(list(state.obse2().values()), dtype='float32')
        next_obse3 = np.array(list(state.obse3().values()), dtype='float32')
        #
        # if not state.enemygroup and state.ourgroup:
        #     reward_o1, reward_o2, reward_o3 = 1, 1, 1
        #     reward_e1, reward_e2, reward_e3 = -0.5, -0.5, -0.5
        #     done = True
        # elif not state.ourgroup and state.enemygroup:
        #     reward_e1, reward_e2, reward_e3 = 1, 1, 1
        #     reward_o1, reward_o2, reward_o3 = -0.5, -0.5, -0.5
        #     done = True
        # elif state.ourgroup and state.enemygroup:
        #     reward_o1, reward_o2, reward_o3 = 0, 0, 0
        #     reward_e1, reward_e2, reward_e3 = 0, 0, 0
        #     done = False
        # else:
        #     reward_o1, reward_o2, reward_o3 = 0, 0, 0
        #     reward_e1, reward_e2, reward_e3 = 0, 0, 0
        #     done = True
        reward_o1, reward_o2, reward_o3, reward_e1, reward_e2, reward_e3 = 0, 0, 0, 0, 0, 0
        if state.our1.health == 0:
            reward_o1 = -1
            done = True
        elif state.our2.health == 0:
            reward_o2 = -1
            done = True
        elif state.our3.health == 0:
            reward_o3 = -1
            done = True
        elif state.enemy1.health == 0:
            reward_e1 = -1
            done = True
        elif state.enemy2.health == 0:
            reward_e2 = -1
            done = True
        elif state.enemy3.health == 0:
            reward_e3 = -1
            done = True
        elif times >= 2000:
            done = True
        else:
            done = False
        times += 1
        obso1 = next_obso1
        obso2 = next_obso2
        obso3 = next_obso3
        obse1 = next_obse1
        obse2 = next_obse2
        obse3 = next_obse3
        # total_reward += reward
        total_rewardo1 += reward_o1
        total_rewardo2 += reward_o2
        total_rewardo3 += reward_o3
        total_rewarde1 += reward_e1
        total_rewarde2 += reward_e2
        total_rewarde3 += reward_e3

        if done:
            break
    return total_rewardo1, total_rewardo2, total_rewardo3, total_rewarde1, total_rewarde2, total_rewarde3


def totrain(screen):
    state = gamestate.GameState(screen)
    train_total_episode = 1e7
    obs_dim = 62
    act_dim = 9
    model1 = AIGameModel(act_dim)
    model2 = AIGameModel(act_dim)
    model3 = AIGameModel(act_dim)
    model4 = AIGameModel(act_dim)
    model5 = AIGameModel(act_dim)
    model6 = AIGameModel(act_dim)
    algorithm1 = parl.algorithms.DDPG(
        model1, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    algorithm2 = parl.algorithms.DDPG(
        model2, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    algorithm3 = parl.algorithms.DDPG(
        model3, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    algorithm4 = parl.algorithms.DDPG(
        model4, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    algorithm5 = parl.algorithms.DDPG(
        model5, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    algorithm6 = parl.algorithms.DDPG(
        model6, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent1 = AIGameAgent(algorithm1, obs_dim, act_dim)
    agent2 = AIGameAgent(algorithm2, obs_dim, act_dim)
    agent3 = AIGameAgent(algorithm3, obs_dim, act_dim)
    agent4 = AIGameAgent(algorithm4, obs_dim, act_dim)
    agent5 = AIGameAgent(algorithm5, obs_dim, act_dim)
    agent6 = AIGameAgent(algorithm6, obs_dim, act_dim)

    agent1.restore('agent1')
    agent2.restore('agent2')
    agent3.restore('agent3')
    agent4.restore('agent4')
    agent5.restore('agent5')
    agent6.restore('agent6')

    rpm = trainingReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    while rpm.size() < MEMORY_WARMUP_SIZE:
        run_train_episode(agent1, agent2, agent3, agent4, agent5, agent6, rpm, screen, state)

    episode = 0
    while episode < train_total_episode:
        for i in range(15):
            train_rewardo1, train_rewardo2, train_rewardo3, train_rewarde1, train_rewarde2, train_rewarde3, = \
                run_train_episode(agent1, agent2, agent3, agent4, agent5, agent6, rpm, screen, state)
            episode += 1
            print('完成了一次比赛')
            logger.info('Episode: {} Reward1: {}'.format(episode, train_rewardo1))
            logger.info('Episode: {} Reward2: {}'.format(episode, train_rewardo2))
            logger.info('Episode: {} Reward3: {}'.format(episode, train_rewardo3))
            logger.info('Episode: {} Reward4: {}'.format(episode, train_rewarde1))
            logger.info('Episode: {} Reward5: {}'.format(episode, train_rewarde2))
            logger.info('Episode: {} Reward6: {}'.format(episode, train_rewarde3))
        agent1.save('agent1')
        agent2.save('agent2')
        agent3.save('agent3')
        agent4.save('agent4')
        agent5.save('agent5')
        agent6.save('agent6')
        print('已保存训练结果')
        evaluate_rewardo1, evaluate_rewardo2, evaluate_rewardo3, evaluate_rewarde1, \
        evaluate_rewarde2, evaluate_rewarde3, = \
            run_evaluate_episode(agent1, agent2, agent3, agent4, agent5, agent6, rpm, screen, state)
        logger.info('Episode {}, Evaluate reward1: {}'.format(episode, evaluate_rewardo1))
        logger.info('Episode {}, Evaluate reward2: {}'.format(episode, evaluate_rewardo2))
        logger.info('Episode {}, Evaluate reward3: {}'.format(episode, evaluate_rewardo3))
        logger.info('Episode {}, Evaluate reward4: {}'.format(episode, evaluate_rewarde1))
        logger.info('Episode {}, Evaluate reward5: {}'.format(episode, evaluate_rewarde2))
        logger.info('Episode {}, Evaluate reward6: {}'.format(episode, evaluate_rewarde3))


class trainingReplayMemory(object):
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obso1 = np.zeros((max_size, obs_dim), dtype='float32')
        self.obso2 = np.zeros((max_size, obs_dim), dtype='float32')
        self.obso3 = np.zeros((max_size, obs_dim), dtype='float32')
        self.obse1 = np.zeros((max_size, obs_dim), dtype='float32')
        self.obse2 = np.zeros((max_size, obs_dim), dtype='float32')
        self.obse3 = np.zeros((max_size, obs_dim), dtype='float32')
        if act_dim == 0:  # Discrete control environment
            self.actiono1 = np.zeros((max_size,), dtype='int32')
            self.actiono2 = np.zeros((max_size,), dtype='int32')
            self.actiono3 = np.zeros((max_size,), dtype='int32')
            self.actione1 = np.zeros((max_size,), dtype='int32')
            self.actione2 = np.zeros((max_size,), dtype='int32')
            self.actione3 = np.zeros((max_size,), dtype='int32')
        else:  # Continuous control environment
            self.actiono1 = np.zeros((max_size, act_dim), dtype='float32')
            self.actiono2 = np.zeros((max_size, act_dim), dtype='float32')
            self.actiono3 = np.zeros((max_size, act_dim), dtype='float32')
            self.actione1 = np.zeros((max_size, act_dim), dtype='float32')
            self.actione2 = np.zeros((max_size, act_dim), dtype='float32')
            self.actione3 = np.zeros((max_size, act_dim), dtype='float32')
        self.rewardo1 = np.zeros((max_size,), dtype='float32')
        self.rewardo2 = np.zeros((max_size,), dtype='float32')
        self.rewardo3 = np.zeros((max_size,), dtype='float32')
        self.rewarde1 = np.zeros((max_size,), dtype='float32')
        self.rewarde2 = np.zeros((max_size,), dtype='float32')
        self.rewarde3 = np.zeros((max_size,), dtype='float32')
        self.terminal = np.zeros((max_size,), dtype='bool')
        self.next_obso1 = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_obso2 = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_obso3 = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_obse1 = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_obse2 = np.zeros((max_size, obs_dim), dtype='float32')
        self.next_obse3 = np.zeros((max_size, obs_dim), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch_o1(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obso1[batch_idx]
        reward = self.rewardo1[batch_idx]
        action = self.actiono1[batch_idx]
        next_obs = self.next_obso1[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def sample_batch_o2(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obso2[batch_idx]
        reward = self.rewardo2[batch_idx]
        action = self.actiono2[batch_idx]
        next_obs = self.next_obso2[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def sample_batch_o3(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obso3[batch_idx]
        reward = self.rewardo3[batch_idx]
        action = self.actiono3[batch_idx]
        next_obs = self.next_obso3[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def sample_batch_e1(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obse1[batch_idx]
        reward = self.rewarde1[batch_idx]
        action = self.actione1[batch_idx]
        next_obs = self.next_obse1[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def sample_batch_e2(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obse2[batch_idx]
        reward = self.rewarde2[batch_idx]
        action = self.actione2[batch_idx]
        next_obs = self.next_obse2[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def sample_batch_e3(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obse3[batch_idx]
        reward = self.rewarde3[batch_idx]
        action = self.actione3[batch_idx]
        next_obs = self.next_obse3[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def make_index(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        return batch_idx

    def sample_batch_by_index(self, batch_idx):
        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def append(self, obs1, act1, reward1, next_obs1, obs2, act2, reward2, next_obs2,
               obs3, act3, reward3, next_obs3, obs4, act4, reward4, next_obs4,
               obs5, act5, reward5, next_obs5, obs6, act6, reward6, next_obs6, terminal):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obso1[self._curr_pos] = obs1
        self.actiono1[self._curr_pos] = act1
        self.rewardo1[self._curr_pos] = reward1
        self.next_obso1[self._curr_pos] = next_obs1
        self.obso2[self._curr_pos] = obs2
        self.actiono2[self._curr_pos] = act2
        self.rewardo2[self._curr_pos] = reward2
        self.next_obso2[self._curr_pos] = next_obs2
        self.obso3[self._curr_pos] = obs3
        self.actiono3[self._curr_pos] = act3
        self.rewardo3[self._curr_pos] = reward3
        self.next_obso3[self._curr_pos] = next_obs3
        self.obse1[self._curr_pos] = obs4
        self.actione1[self._curr_pos] = act4
        self.rewarde1[self._curr_pos] = reward4
        self.next_obse1[self._curr_pos] = next_obs4
        self.obse2[self._curr_pos] = obs5
        self.actione2[self._curr_pos] = act5
        self.rewarde2[self._curr_pos] = reward5
        self.next_obse2[self._curr_pos] = next_obs5
        self.obse3[self._curr_pos] = obs6
        self.actione3[self._curr_pos] = act6
        self.rewarde3[self._curr_pos] = reward6
        self.next_obse3[self._curr_pos] = next_obs6
        self.terminal[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size

    def save(self, pathname):
        other = np.array([self._curr_size, self._curr_pos], dtype=np.int32)
        np.savez(
            pathname,
            obso1=self.obso1,
            actiono1=self.actiono1,
            rewardo1=self.rewardo1,
            next_obso1=self.next_obso1,
            obso2=self.obso2,
            actiono2=self.actiono2,
            rewardo2=self.rewardo2,
            next_obso2=self.next_obso2,
            obso3=self.obso3,
            actiono3=self.actiono3,
            rewaro3=self.rewardo3,
            next_obso3=self.next_obso3,
            obse1=self.obse1,
            actione1=self.actione1,
            rewarde1=self.rewarde1,
            next_obse1=self.next_obse1,
            obse2=self.obse2,
            actione2=self.actione2,
            rewarde2=self.rewarde2,
            next_obse2=self.next_obse2,
            obse3=self.obse3,
            actione3=self.actione3,
            rewarde3=self.rewarde3,
            next_obse3=self.next_obse3,
            terminal=self.terminal,
            other=other)

    def load(self, pathname):
        data = np.load(pathname)
        other = data['other']
        if int(other[0]) > self.max_size:
            logger.warn('loading from a bigger size rpm!')
        self._curr_size = min(int(other[0]), self.max_size)
        self._curr_pos = min(int(other[1]), self.max_size - 1)

        self.obso1[:self._curr_size] = data['obso1'][:self._curr_size]
        self.actiono1[:self._curr_size] = data['actiono1'][:self._curr_size]
        self.rewardo1[:self._curr_size] = data['rewardo1'][:self._curr_size]
        self.next_obso1[:self._curr_size] = data['next_obso1'][:self._curr_size]
        self.obso2[:self._curr_size] = data['obso2'][:self._curr_size]
        self.actiono2[:self._curr_size] = data['actiono2'][:self._curr_size]
        self.rewardo2[:self._curr_size] = data['rewardo2'][:self._curr_size]
        self.next_obso2[:self._curr_size] = data['next_obso2'][:self._curr_size]
        self.obso3[:self._curr_size] = data['obso3'][:self._curr_size]
        self.actiono3[:self._curr_size] = data['actiono3'][:self._curr_size]
        self.rewardo3[:self._curr_size] = data['rewardo3'][:self._curr_size]
        self.next_obso3[:self._curr_size] = data['next_obso3'][:self._curr_size]
        self.obse1[:self._curr_size] = data['obse1'][:self._curr_size]
        self.actione1[:self._curr_size] = data['actione1'][:self._curr_size]
        self.rewarde1[:self._curr_size] = data['rewarde1'][:self._curr_size]
        self.next_obse1[:self._curr_size] = data['next_obse1'][:self._curr_size]
        self.obse2[:self._curr_size] = data['obse2'][:self._curr_size]
        self.actione2[:self._curr_size] = data['actione2'][:self._curr_size]
        self.rewarde2[:self._curr_size] = data['rewarde2'][:self._curr_size]
        self.next_obse2[:self._curr_size] = data['next_obse2'][:self._curr_size]
        self.obse3[:self._curr_size] = data['obse3'][:self._curr_size]
        self.actione3[:self._curr_size] = data['actione3'][:self._curr_size]
        self.rewarde3[:self._curr_size] = data['rewarde3'][:self._curr_size]
        self.next_obse3[:self._curr_size] = data['next_obse3'][:self._curr_size]
        self.terminal[:self._curr_size] = data['terminal'][:self._curr_size]
        logger.info("[load rpm]memory loade from {}".format(pathname))


if __name__ == 'main':
    print('helloworld!')
