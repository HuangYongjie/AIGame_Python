import pygame
import math
import numpy as np


def newadd(amounta, amountb, top):
    if amounta + amountb < top:
        return amounta + amountb
    else:
        return top


def newmin(a, b, butt):
    if a - b > butt:
        return a - b
    else:
        return butt


def toangdir(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:
        if dy == 0:
            ang = None
        elif dy > 0:
            ang = 0.5 * math.pi
        else:
            ang = 1.5 * math.pi
    else:
        an = math.atan(dy / dx)
        if dx > 0:
            ang = an
        else:
            ang = math.pi + an
    return ang


def todirxy(direc):
    return math.cos(direc), math.sin(direc)


MAXMOVESPEED = 4
BASICSKILLSPEED = 8


class Creatures(pygame.sprite.Sprite):
    def __init__(self, color, width, height, posX, posY, moveareax, moveareay, ID, talent1=5, talent2=5, talent3=5):
        pygame.sprite.Sprite.__init__(self)
        self.isalive = True
        self.font = pygame.freetype.Font(r"C:\Windows\Fonts\simsun.ttc", 20)
        self.ID = ID
        self.width = width
        self.height = height
        self.moveareax, self.moveareay = [moveareax - self.width, moveareay - self.height]
        self.image = pygame.Surface([self.width, self.height])
        self.image.fill(color)
        self.speed = [0., 0.]
        self.health = 100.
        self.energy = 50.
        self.buff = 0.
        self.talent1, self.talent2, self.talent3 = [talent1, talent2, talent3]
        self.position = [posX, posY]
        self.skillstorage = 2
        self.rect = self.image.get_rect()
        self.sencerect = pygame.Rect(self.position[0] - 3 * self.width,
                                     self.position[1] - 3 * self.height,
                                     self.width * 7,
                                     self.height * 7)
        self.healthbarimage = pygame.Surface([self.width * 0.8 * self.health / 100, self.height * 0.1])
        self.healthbarimage.fill([255, 0, 0])
        self.energybarimage = pygame.Surface([self.width * 0.8 * self.energy / 100, self.height * 0.1])
        self.energybarimage.fill([0, 0, 255])
        self.healthbar = pygame.Rect(self.position[0] + 0.1 * self.width,
                                     self.position[1] + 0.1 * self.height,
                                     self.width * 0.8 * self.health / 100,
                                     self.height * 0.1)
        self.energybar = pygame.Rect(self.position[0] + 0.1 * self.width,
                                     self.position[1] + 0.3 * self.height,
                                     self.width * 0.8 * self.energy / 100,
                                     self.height * 0.1)

    def reset(self, pos):
        self.isalive = True
        self.health = 100.
        self.energy = 50.
        self.buff = 0.
        self.speed = [0., 0.]
        self.skillstorage = 2
        self.position = pos

    # 设定buff值如何影响自身技能威力
    def buffform(self):
        return self.buff * 0.01 + 1

    def skill(self, skillgroup, skilltype, direction, strength=0):
        if self.skillstorage >= 1:
            if strength < 0:
                strength = 0
            if self.energy < 100 * strength:
                strength = self.energy / 100
                self.energy = 0.
            else:
                self.energy -= 100 * strength
            spd = BASICSKILLSPEED * (1 + strength)
            directionx, directiony = todirxy(2 * math.pi * direction)
            spdx = spd * directionx
            spdy = spd * directiony
            startdistance = 0.5 * (self.height + self.width)
            if skilltype == 'skill1':
                # print('{}朝{}方向发出了技能一'.format(self.ID, direction))
                SkillSprite(color='red',
                            pos=[self.position[0] + 0.5 * self.width + startdistance * directionx,
                                 self.position[1] + 0.5 * self.height + startdistance * directiony],
                            speed=[spdx, spdy], power=3 * self.talent1 * self.buffform() * (1 + strength),
                            groups=skillgroup)
            elif skilltype == 'skill2':
                SkillSprite(color='green',
                            pos=[self.position[0] + 0.5 * self.width + startdistance * directionx,
                                 self.position[1] + 0.5 * self.height + startdistance * directiony],
                            speed=[spdx, spdy], power=3 * self.talent2 * self.buffform() * (1 + strength),
                            groups=skillgroup)
            elif skilltype == 'skill3':
                SkillSprite(color='blue',
                            pos=[self.position[0] + 0.5 * self.width + startdistance * directionx,
                                 self.position[1] + 0.5 * self.height + startdistance * directiony],
                            speed=[spdx, spdy], power=3 * self.talent3 * self.buffform() * (1 + strength),
                            groups=skillgroup)
            elif skilltype == 'skill4':
                SkillSprite(color='orange',
                            pos=[self.position[0] + 0.5 * self.width + startdistance * directionx,
                                 self.position[1] + 0.5 * self.height + startdistance * directiony],
                            speed=[spdx, spdy], power=3 * self.talent3 * self.buffform() * (1 + strength),
                            groups=skillgroup)
            self.skillstorage -= 1

    # 检测附近的技能弹射，返回是否有技能弹射、技能弹射位置、速度
    def skdetect(self, skillgroup):
        detectedsk = []
        for each in skillgroup:
            if self.sencerect.colliderect(each.rect):
                detectedsk.append(each)
        if detectedsk:
            # print('{}感知到了技能'.format(self.ID))
            return True, detectedsk[0].position[0], detectedsk[0].position[1], detectedsk[0].speed[0], detectedsk[0].speed[1]
        else:
            return False, 0, 0, 0, 0

    # 检测自身是否被击中，若被击中，执行被击中操作，并删除击中自己的弹射
    def skhitted(self, skillgroup, sktype):
        for each in skillgroup:
            if pygame.sprite.collide_rect(self, each):
                self.hitted(sktype, each.power)
                each.kill()

    def hitted(self, sktype, skpower):
        if sktype == 'skill1':  # 技能一设定为伤害技能，按照技能威力减少目标血量，并增加目标少量能量
            self.health = newmin(self.health, skpower, 0)
            self.energy = newadd(self.energy, 5, 100)
            print('{}被技能一命中，失去{:.2f}生命，还有{:.2f}生命'.format(self.ID, skpower, self.health))
        if sktype == 'skill2':  # 技能二设定为治疗技能，按照技能威力增加目标血量，并增加一定能量
            self.health = newadd(self.health, skpower, 100)
            self.energy = newadd(self.energy, 10, 100)
        if sktype == 'skill3':  # 技能三、技能四设定为辅助技能，按照技能威力改变buff值和能量值
            self.buff = newmin(self.buff, skpower, -99)
            self.energy = newmin(self.energy, skpower, 0)
        if sktype == 'skill4':
            self.buff = newadd(self.buff, skpower, 99)
            self.energy = newadd(self.energy, skpower, 100)

    #
    def update(self, screen, sk1group, sk2group, sk3group, sk4group, chanceofnoskill,
               chanceofskill1, chanceofskill2, chanceofskill3, chanceofskill4,
               skilldirect, skillstrength, movedirect, movestrength):
        if self.isalive:
            whitchskill = np.array([chanceofnoskill, chanceofskill1, chanceofskill2, chanceofskill3, chanceofskill4])
            chance = np.max(whitchskill)
            sktp = np.random.choice(np.where(whitchskill == chance)[0])
            if sktp == 1:
                self.skill(sk1group, 'skill1', skilldirect, skillstrength)
            elif sktp == 2:
                self.skill(sk2group, 'skill2', skilldirect, skillstrength)
            elif sktp == 3:
                self.skill(sk3group, 'skill3', skilldirect, skillstrength)
            elif sktp == 4:
                self.skill(sk4group, 'skill4', skilldirect, skillstrength)
            self.move(movedirect, movestrength)
            self.skhitted(sk1group, sktype='skill1')
            self.skhitted(sk2group, sktype='skill2')
            self.skhitted(sk3group, sktype='skill3')
            self.skhitted(sk4group, sktype='skill4')
            self.skillstorage = newadd(self.skillstorage, 0.08, 3)
            self.energy = newadd(self.energy, 0.1, 100)
            if self.buff < 0:
                self.buff = newadd(self.buff, 0.3, 0)
            else:
                self.buff = newmin(self.buff, 0.3, 0)
            self.rect.left = self.position[0]
            self.rect.top = self.position[1]
            self.healthbarimage = pygame.Surface([self.width * 0.8 * self.health / 100, self.height * 0.1])
            self.healthbarimage.fill([255, 0, 0])
            self.energybarimage = pygame.Surface([self.width * 0.8 * self.energy / 100, self.height * 0.1])
            self.energybarimage.fill([0, 0, 255])
            self.font.render_to(self.image, [self.width / 10, 5 * self.width / 10], self.ID, fgcolor=[0, 0, 0])
            screen.blit(self.image, self.rect)
            screen.blit(self.healthbarimage, self.healthbar)
            screen.blit(self.energybarimage, self.energybar)
            if self.health <= 0:
                self.isalive = False
                self.kill()
                print('{}已死亡'.format(self.ID))

    # 下面是计算速度以及位置变化的函数
    # 想要的效果：朝所有方向的运动规律相同，在当前花费能量值一定的情况下，速度分布在一个圆形内，花费能量可以增加加速度，增加速度上限，
    # 花费高能量能短时间转向：0.5秒速度变化大约3个初始速度上限值，手动操作具有一定的手感，即不花费能量时转向不宜太僵硬：1秒变化一个初始速度上限
    # 计算式推导的基本逻辑是：先根据参数计算出变化后的速度，如果计算出的速度超过设置的上限，则减小速度大小到等于上限而保持速度方向不变
    def move(self, direction=None, strength=0):
        if direction is not None:
            direction *= 2 * math.pi
            # 先计算每帧预期能量花费
            if strength < 0.75:
                cost = 0
            elif strength < 1:
                cost = 4 * (strength - 0.75)
            else:
                cost = 1
            # 设定在此能量花费下的速度绝对值变化量、速度上限
            acc = (1 + 6 * cost) * MAXMOVESPEED
            vmax =  MAXMOVESPEED * (1 + cost)
            v1 = [self.speed[0] + math.cos(direction) * acc, self.speed[1] + math.sin(direction) * acc]
            # if v1[0] < 0 and abs(v1[1]) < 0.0000001:
            #     print('正在朝左加速')
            # elif v1[0] > 0 and abs(v1[1]) < 0.0000001:
            #     print('正在朝右加速')
            # elif abs(v1[0]) < 0.0000001 and v1[1] > 0:
            #     print('正在朝下加速')
            # elif abs(v1[0]) < 0.0000001 and v1[1] < 0:
            #     print('正在朝上加速')
            # 限制速度不超过上限
            if v1[0] * v1[0] + v1[1] * v1[1] > vmax * vmax:
                a = vmax / math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
                v1[0] = v1[0] * a
                v1[1] = v1[1] * a
                deltav = math.sqrt((self.speed[0] - v1[0]) * (self.speed[0] - v1[0])
                                   + (self.speed[0] - v1[1]) * (self.speed[0] - v1[1]))
                # print(deltav)
                # print(MAXMOVESPEED * 0.03)
                if deltav > MAXMOVESPEED:
                    cost = ((deltav / MAXMOVESPEED) - 1) / 6
                    # print('deltav too much')
                else:
                    cost = 0
            self.energy = newmin(self.energy, cost, 0)
            self.speed[0] = v1[0]
            self.speed[1] = v1[1]
        # 触壁反弹
        if self.position[0] < 0 or self.position[0] > self.moveareax:
            self.speed[0] = -self.speed[0]
            if self.position[0] < 0:
                self.position[0] = -self.position[0]
            elif self.position[0] > self.moveareax:
                self.position[0] = 2 * self.moveareax - self.position[0]
        if self.position[1] < 0 or self.position[1] > self.moveareay:
            self.speed[1] = -self.speed[1]
            if self.position[1] < 0:
                self.position[1] = -self.position[1]
            elif self.position[1] > self.moveareay:
                self.position[1] = 2 * self.moveareay - self.position[1]
        # 摩擦
        self.speed[0] *= 0.976
        self.speed[1] *= 0.976
        if abs(self.speed[0]) + abs(self.speed[1]) < 0.9:
            self.speed = [0., 0.]
        self.position[0] += self.speed[0]
        self.position[1] += self.speed[1]
        self.rect.left = self.position[0]
        self.rect.top = self.position[1]
        self.sencerect.left = self.position[0] - 3 * self.width
        self.sencerect.top = self.position[1] - 3 * self.height
        self.healthbar.left = self.position[0] + 0.1 * self.width
        self.healthbar.top = self.position[1] + 0.1 * self.height
        self.healthbar.width = self.width * 0.8 * self.health / 100
        self.energybar.left = self.position[0] + 0.1 * self.width
        self.energybar.top = self.position[1] + 0.3 * self.height
        self.energybar.width = self.width * 0.8 * self.energy / 100


class SkillSprite(pygame.sprite.Sprite):
    def __init__(self, color, pos, speed, power, groups):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([4, 4])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.position = pos
        self.speed = speed
        self.power = power
        self.reflectcount = 0
        groups.add(self)

    def update(self, screen):
        if self.position[0] < 0 or self.position[0] > screen.get_width():
            if self.reflectcount == 1:
                self.speed[0] = -self.speed[0]
                self.reflectcount -= 1
            else:
                self.kill()
        if self.position[1] < 0 or self.position[1] > screen.get_height():
            if self.reflectcount == 1:
                self.speed[1] = -self.speed[1]
                self.reflectcount -= 1
            else:
                self.kill()
        self.position[0] += self.speed[0]
        self.position[1] += self.speed[1]
        self.rect.left = self.position[0]
        self.rect.top = self.position[1]
        screen.blit(self.image, self.rect)
