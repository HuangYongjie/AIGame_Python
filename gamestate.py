import pygame
import creatures
import random


class GameState:
    def __init__(self, screen):
        self.width, self.height = screen.get_width(), screen.get_height()
        self.our1 = creatures.Creatures(color=[200, 150, 150],
                                        width=50, height=50, moveareax=self.width, moveareay=self.height,
                                        posX=random.random() * (self.width - 50),
                                        posY=random.random() * (self.height - 50),
                                        ID='A1', talent1=999)

        self.our2 = creatures.Creatures(color=[150, 200, 150],
                                        width=50, height=50, moveareax=self.width, moveareay=self.height,
                                        posX=random.random() * (self.width - 50),
                                        posY=random.random() * (self.height - 50),
                                        ID='A2')
        self.our3 = creatures.Creatures(color=[150, 150, 200],
                                        width=50, height=50, moveareax=self.width, moveareay=self.height,
                                        posX=random.random() * (self.width - 50),
                                        posY=random.random() * (self.height - 50),
                                        ID='A3')
        self.enemy1 = creatures.Creatures(color=[55, 80, 80],
                                          width=50, height=50, moveareax=self.width, moveareay=self.height,
                                          posX=random.random() * (self.width - 50),
                                          posY=random.random() * (self.height - 50),
                                          ID='B1')
        self.enemy2 = creatures.Creatures(color=[80, 55, 80],
                                          width=50, height=50, moveareax=self.width, moveareay=self.height,
                                          posX=random.random() * (self.width - 50),
                                          posY=random.random() * (self.height - 50),
                                          ID='B2')
        self.enemy3 = creatures.Creatures(color=[80, 80, 55],
                                          width=50, height=50, moveareax=self.width, moveareay=self.height,
                                          posX=random.random() * (self.width - 50),
                                          posY=random.random() * (self.height - 50),
                                          ID='B3')
        self.ourgroup = pygame.sprite.Group()
        self.ourgroup.add(self.our1)
        self.ourgroup.add(self.our2)
        self.ourgroup.add(self.our3)
        self.enemygroup = pygame.sprite.Group()
        self.enemygroup.add(self.enemy1)
        self.enemygroup.add(self.enemy2)
        self.enemygroup.add(self.enemy3)
        self.sk1group = pygame.sprite.Group()
        self.sk2group = pygame.sprite.Group()
        self.sk3group = pygame.sprite.Group()
        self.sk4group = pygame.sprite.Group()

    def obso1(self):
        sk1detarg = self.our1.skdetect(self.sk1group)
        sk2detarg = self.our1.skdetect(self.sk2group)
        sk3detarg = self.our1.skdetect(self.sk3group)
        sk4detarg = self.our1.skdetect(self.sk4group)
        obs = {'myposx': self.our1.position[0],
               'myposy': self.our1.position[1],
               'myspeedx': self.our1.speed[0],
               'myspeedy': self.our1.speed[1],
               'myhealth': self.our1.health,
               'myenergy': self.our1.energy,
               'mybuff': self.our1.buff,
               'myskillstorage': self.our1.skillstorage,
               'ally1posx': self.our2.position[0],
               'ally1posy': self.our2.position[1],
               'ally1speedx': self.our2.speed[0],
               'ally1speedy': self.our2.speed[1],
               'ally1health': self.our2.health,
               'ally1energy': self.our2.energy,
               'ally1buff': self.our2.buff,
               'ally1skillstorage': self.our2.skillstorage,
               'ally2posx': self.our3.position[0],
               'ally2posy': self.our3.position[1],
               'ally2speedx': self.our3.speed[0],
               'ally2speedy': self.our3.speed[1],
               'ally2health': self.our3.health,
               'ally2energy': self.our3.energy,
               'ally2buff': self.our3.buff,
               'ally2skillstorage': self.our3.skillstorage,
               'enemy1posx': self.enemy1.position[0],
               'enemy1posy': self.enemy1.position[1],
               'enemy1speedx': self.enemy1.speed[0],
               'enemy1speedy': self.enemy1.speed[1],
               'enemy1health': self.enemy1.health,
               'enemy1energy': self.enemy1.energy,
               'enemy2posx': self.enemy2.position[0],
               'enemy2posy': self.enemy2.position[1],
               'enemy2speedx': self.enemy2.speed[0],
               'enemy2speedy': self.enemy2.speed[1],
               'enemy2health': self.enemy2.health,
               'enemy2energy': self.enemy2.energy,
               'enemy3posx': self.enemy3.position[0],
               'enemy3posy': self.enemy3.position[1],
               'enemy3speedx': self.enemy3.speed[0],
               'enemy3speedy': self.enemy3.speed[1],
               'enemy3health': self.enemy3.health,
               'enemy3energy': self.enemy3.energy,
               'sk1exist': sk1detarg[0],
               'sk1posx': sk1detarg[1],
               'sk1posy': sk1detarg[2],
               'sk1speedx': sk1detarg[3],
               'sk1speedy': sk1detarg[4],
               'sk2exist': sk2detarg[0],
               'sk2posx': sk2detarg[1],
               'sk2posy': sk2detarg[2],
               'sk2speedx': sk2detarg[3],
               'sk2speedy': sk2detarg[4],
               'sk3exist': sk3detarg[0],
               'sk3posx': sk3detarg[1],
               'sk3posy': sk3detarg[2],
               'sk3speedx': sk3detarg[3],
               'sk3speedy': sk3detarg[4],
               'sk4exist': sk4detarg[0],
               'sk4posx': sk4detarg[1],
               'sk4posy': sk4detarg[2],
               'sk4speedx': sk4detarg[3],
               'sk4speedy': sk4detarg[4]}
        return obs

    def obso2(self):
        sk1detarg = self.our2.skdetect(self.sk1group)
        sk2detarg = self.our2.skdetect(self.sk2group)
        sk3detarg = self.our2.skdetect(self.sk3group)
        sk4detarg = self.our2.skdetect(self.sk4group)
        obs = {'myposx': self.our2.position[0],
               'myposy': self.our2.position[1],
               'myspeedx': self.our2.speed[0],
               'myspeedy': self.our2.speed[1],
               'myhealth': self.our2.health,
               'myenergy': self.our2.energy,
               'mybuff': self.our2.buff,
               'myskillstorage': self.our2.skillstorage,
               'ally1posx': self.our1.position[0],
               'ally1posy': self.our1.position[1],
               'ally1speedx': self.our1.speed[0],
               'ally1speedy': self.our1.speed[1],
               'ally1health': self.our1.health,
               'ally1energy': self.our1.energy,
               'ally1buff': self.our1.buff,
               'ally1skillstorage': self.our1.skillstorage,
               'ally2posx': self.our3.position[0],
               'ally2posy': self.our3.position[1],
               'ally2speedx': self.our3.speed[0],
               'ally2speedy': self.our3.speed[1],
               'ally2health': self.our3.health,
               'ally2energy': self.our3.energy,
               'ally2buff': self.our3.buff,
               'ally2skillstorage': self.our3.skillstorage,
               'enemy1posx': self.enemy1.position[0],
               'enemy1posy': self.enemy1.position[1],
               'enemy1speedx': self.enemy1.speed[0],
               'enemy1speedy': self.enemy1.speed[1],
               'enemy1health': self.enemy1.health,
               'enemy1energy': self.enemy1.energy,
               'enemy2posx': self.enemy2.position[0],
               'enemy2posy': self.enemy2.position[1],
               'enemy2speedx': self.enemy2.speed[0],
               'enemy2speedy': self.enemy2.speed[1],
               'enemy2health': self.enemy2.health,
               'enemy2energy': self.enemy2.energy,
               'enemy3posx': self.enemy3.position[0],
               'enemy3posy': self.enemy3.position[1],
               'enemy3speedx': self.enemy3.speed[0],
               'enemy3speedy': self.enemy3.speed[1],
               'enemy3health': self.enemy3.health,
               'enemy3energy': self.enemy3.energy,
               'sk1exist': sk1detarg[0],
               'sk1posx': sk1detarg[1],
               'sk1posy': sk1detarg[2],
               'sk1speedx': sk1detarg[3],
               'sk1speedy': sk1detarg[4],
               'sk2exist': sk2detarg[0],
               'sk2posx': sk2detarg[1],
               'sk2posy': sk2detarg[2],
               'sk2speedx': sk2detarg[3],
               'sk2speedy': sk2detarg[4],
               'sk3exist': sk3detarg[0],
               'sk3posx': sk3detarg[1],
               'sk3posy': sk3detarg[2],
               'sk3speedx': sk3detarg[3],
               'sk3speedy': sk3detarg[4],
               'sk4exist': sk4detarg[0],
               'sk4posx': sk4detarg[1],
               'sk4posy': sk4detarg[2],
               'sk4speedx': sk4detarg[3],
               'sk4speedy': sk4detarg[4]}
        return obs

    def obso3(self):
        sk1detarg = self.our3.skdetect(self.sk1group)
        sk2detarg = self.our3.skdetect(self.sk2group)
        sk3detarg = self.our3.skdetect(self.sk3group)
        sk4detarg = self.our3.skdetect(self.sk4group)
        obs = {'myposx': self.our3.position[0],
               'myposy': self.our3.position[1],
               'myspeedx': self.our3.speed[0],
               'myspeedy': self.our3.speed[1],
               'myhealth': self.our3.health,
               'myenergy': self.our3.energy,
               'mybuff': self.our3.buff,
               'myskillstorage': self.our3.skillstorage,
               'ally1posx': self.our2.position[0],
               'ally1posy': self.our2.position[1],
               'ally1speedx': self.our2.speed[0],
               'ally1speedy': self.our2.speed[1],
               'ally1health': self.our2.health,
               'ally1energy': self.our2.energy,
               'ally1buff': self.our2.buff,
               'ally1skillstorage': self.our2.skillstorage,
               'ally2posx': self.our1.position[0],
               'ally2posy': self.our1.position[1],
               'ally2speedx': self.our1.speed[0],
               'ally2speedy': self.our1.speed[1],
               'ally2health': self.our1.health,
               'ally2energy': self.our1.energy,
               'ally2buff': self.our1.buff,
               'ally2skillstorage': self.our1.skillstorage,
               'enemy1posx': self.enemy1.position[0],
               'enemy1posy': self.enemy1.position[1],
               'enemy1speedx': self.enemy1.speed[0],
               'enemy1speedy': self.enemy1.speed[1],
               'enemy1health': self.enemy1.health,
               'enemy1energy': self.enemy1.energy,
               'enemy2posx': self.enemy2.position[0],
               'enemy2posy': self.enemy2.position[1],
               'enemy2speedx': self.enemy2.speed[0],
               'enemy2speedy': self.enemy2.speed[1],
               'enemy2health': self.enemy2.health,
               'enemy2energy': self.enemy2.energy,
               'enemy3posx': self.enemy3.position[0],
               'enemy3posy': self.enemy3.position[1],
               'enemy3speedx': self.enemy3.speed[0],
               'enemy3speedy': self.enemy3.speed[1],
               'enemy3health': self.enemy3.health,
               'enemy3energy': self.enemy3.energy,
               'sk1exist': sk1detarg[0],
               'sk1posx': sk1detarg[1],
               'sk1posy': sk1detarg[2],
               'sk1speedx': sk1detarg[3],
               'sk1speedy': sk1detarg[4],
               'sk2exist': sk2detarg[0],
               'sk2posx': sk2detarg[1],
               'sk2posy': sk2detarg[2],
               'sk2speedx': sk2detarg[3],
               'sk2speedy': sk2detarg[4],
               'sk3exist': sk3detarg[0],
               'sk3posx': sk3detarg[1],
               'sk3posy': sk3detarg[2],
               'sk3speedx': sk3detarg[3],
               'sk3speedy': sk3detarg[4],
               'sk4exist': sk4detarg[0],
               'sk4posx': sk4detarg[1],
               'sk4posy': sk4detarg[2],
               'sk4speedx': sk4detarg[3],
               'sk4speedy': sk4detarg[4]}
        return obs

    def obse1(self):
        sk1detarg = self.enemy1.skdetect(self.sk1group)
        sk2detarg = self.enemy1.skdetect(self.sk2group)
        sk3detarg = self.enemy1.skdetect(self.sk3group)
        sk4detarg = self.enemy1.skdetect(self.sk4group)
        obs = {'myposx': self.enemy1.position[0],
               'myposy': self.enemy1.position[1],
               'myspeedx': self.enemy1.speed[0],
               'myspeedy': self.enemy1.speed[1],
               'myhealth': self.enemy1.health,
               'myenergy': self.enemy1.energy,
               'mybuff': self.enemy1.buff,
               'myskillstorage': self.enemy1.skillstorage,
               'ally1posx': self.enemy2.position[0],
               'ally1posy': self.enemy2.position[1],
               'ally1speedx': self.enemy2.speed[0],
               'ally1speedy': self.enemy2.speed[1],
               'ally1health': self.enemy2.health,
               'ally1energy': self.enemy2.energy,
               'ally1buff': self.enemy2.buff,
               'ally1skillstorage': self.enemy2.skillstorage,
               'ally2posx': self.enemy3.position[0],
               'ally2posy': self.enemy3.position[1],
               'ally2speedx': self.enemy3.speed[0],
               'ally2speedy': self.enemy3.speed[1],
               'ally2health': self.enemy3.health,
               'ally2energy': self.enemy3.energy,
               'ally2buff': self.enemy3.buff,
               'ally2skillstorage': self.enemy3.skillstorage,
               'enemy1posx': self.our1.position[0],
               'enemy1posy': self.our1.position[1],
               'enemy1speedx': self.our1.speed[0],
               'enemy1speedy': self.our1.speed[1],
               'enemy1health': self.our1.health,
               'enemy1energy': self.our1.energy,
               'enemy2posx': self.our2.position[0],
               'enemy2posy': self.our2.position[1],
               'enemy2speedx': self.our2.speed[0],
               'enemy2speedy': self.our2.speed[1],
               'enemy2health': self.our2.health,
               'enemy2energy': self.our2.energy,
               'enemy3posx': self.our3.position[0],
               'enemy3posy': self.our3.position[1],
               'enemy3speedx': self.our3.speed[0],
               'enemy3speedy': self.our3.speed[1],
               'enemy3health': self.our3.health,
               'enemy3energy': self.our3.energy,
               'sk1exist': sk1detarg[0],
               'sk1posx': sk1detarg[1],
               'sk1posy': sk1detarg[2],
               'sk1speedx': sk1detarg[3],
               'sk1speedy': sk1detarg[4],
               'sk2exist': sk2detarg[0],
               'sk2posx': sk2detarg[1],
               'sk2posy': sk2detarg[2],
               'sk2speedx': sk2detarg[3],
               'sk2speedy': sk2detarg[4],
               'sk3exist': sk3detarg[0],
               'sk3posx': sk3detarg[1],
               'sk3posy': sk3detarg[2],
               'sk3speedx': sk3detarg[3],
               'sk3speedy': sk3detarg[4],
               'sk4exist': sk4detarg[0],
               'sk4posx': sk4detarg[1],
               'sk4posy': sk4detarg[2],
               'sk4speedx': sk4detarg[3],
               'sk4speedy': sk4detarg[4]}
        return obs

    def obse2(self):
        sk1detarg = self.enemy2.skdetect(self.sk1group)
        sk2detarg = self.enemy2.skdetect(self.sk2group)
        sk3detarg = self.enemy2.skdetect(self.sk3group)
        sk4detarg = self.enemy2.skdetect(self.sk4group)
        obs = {'myposx': self.enemy2.position[0],
               'myposy': self.enemy2.position[1],
               'myspeedx': self.enemy2.speed[0],
               'myspeedy': self.enemy2.speed[1],
               'myhealth': self.enemy2.health,
               'myenergy': self.enemy2.energy,
               'mybuff': self.enemy2.buff,
               'myskillstorage': self.enemy2.skillstorage,
               'ally1posx': self.enemy1.position[0],
               'ally1posy': self.enemy1.position[1],
               'ally1speedx': self.enemy1.speed[0],
               'ally1speedy': self.enemy1.speed[1],
               'ally1health': self.enemy1.health,
               'ally1energy': self.enemy1.energy,
               'ally1buff': self.enemy1.buff,
               'ally1skillstorage': self.enemy1.skillstorage,
               'ally2posx': self.enemy3.position[0],
               'ally2posy': self.enemy3.position[1],
               'ally2speedx': self.enemy3.speed[0],
               'ally2speedy': self.enemy3.speed[1],
               'ally2health': self.enemy3.health,
               'ally2energy': self.enemy3.energy,
               'ally2buff': self.enemy3.buff,
               'ally2skillstorage': self.enemy3.skillstorage,
               'enemy1posx': self.our1.position[0],
               'enemy1posy': self.our1.position[1],
               'enemy1speedx': self.our1.speed[0],
               'enemy1speedy': self.our1.speed[1],
               'enemy1health': self.our1.health,
               'enemy1energy': self.our1.energy,
               'enemy2posx': self.our2.position[0],
               'enemy2posy': self.our2.position[1],
               'enemy2speedx': self.our2.speed[0],
               'enemy2speedy': self.our2.speed[1],
               'enemy2health': self.our2.health,
               'enemy2energy': self.our2.energy,
               'enemy3posx': self.our3.position[0],
               'enemy3posy': self.our3.position[1],
               'enemy3speedx': self.our3.speed[0],
               'enemy3speedy': self.our3.speed[1],
               'enemy3health': self.our3.health,
               'enemy3energy': self.our3.energy,
               'sk1exist': sk1detarg[0],
               'sk1posx': sk1detarg[1],
               'sk1posy': sk1detarg[2],
               'sk1speedx': sk1detarg[3],
               'sk1speedy': sk1detarg[4],
               'sk2exist': sk2detarg[0],
               'sk2posx': sk2detarg[1],
               'sk2posy': sk2detarg[2],
               'sk2speedx': sk2detarg[3],
               'sk2speedy': sk2detarg[4],
               'sk3exist': sk3detarg[0],
               'sk3posx': sk3detarg[1],
               'sk3posy': sk3detarg[2],
               'sk3speedx': sk3detarg[3],
               'sk3speedy': sk3detarg[4],
               'sk4exist': sk4detarg[0],
               'sk4posx': sk4detarg[1],
               'sk4posy': sk4detarg[2],
               'sk4speedx': sk4detarg[3],
               'sk4speedy': sk4detarg[4]}
        return obs

    def obse3(self):
        sk1detarg = self.enemy3.skdetect(self.sk1group)
        sk2detarg = self.enemy3.skdetect(self.sk2group)
        sk3detarg = self.enemy3.skdetect(self.sk3group)
        sk4detarg = self.enemy3.skdetect(self.sk4group)
        obs = {'myposx': self.enemy3.position[0],
               'myposy': self.enemy3.position[1],
               'myspeedx': self.enemy3.speed[0],
               'myspeedy': self.enemy3.speed[1],
               'myhealth': self.enemy3.health,
               'myenergy': self.enemy3.energy,
               'mybuff': self.enemy3.buff,
               'myskillstorage': self.enemy3.skillstorage,
               'ally1posx': self.enemy2.position[0],
               'ally1posy': self.enemy2.position[1],
               'ally1speedx': self.enemy2.speed[0],
               'ally1speedy': self.enemy2.speed[1],
               'ally1health': self.enemy2.health,
               'ally1energy': self.enemy2.energy,
               'ally1buff': self.enemy2.buff,
               'ally1skillstorage': self.enemy2.skillstorage,
               'ally2posx': self.enemy1.position[0],
               'ally2posy': self.enemy1.position[1],
               'ally2speedx': self.enemy1.speed[0],
               'ally2speedy': self.enemy1.speed[1],
               'ally2health': self.enemy1.health,
               'ally2energy': self.enemy1.energy,
               'ally2buff': self.enemy1.buff,
               'ally2skillstorage': self.enemy1.skillstorage,
               'enemy1posx': self.our1.position[0],
               'enemy1posy': self.our1.position[1],
               'enemy1speedx': self.our1.speed[0],
               'enemy1speedy': self.our1.speed[1],
               'enemy1health': self.our1.health,
               'enemy1energy': self.our1.energy,
               'enemy2posx': self.our2.position[0],
               'enemy2posy': self.our2.position[1],
               'enemy2speedx': self.our2.speed[0],
               'enemy2speedy': self.our2.speed[1],
               'enemy2health': self.our2.health,
               'enemy2energy': self.our2.energy,
               'enemy3posx': self.our3.position[0],
               'enemy3posy': self.our3.position[1],
               'enemy3speedx': self.our3.speed[0],
               'enemy3speedy': self.our3.speed[1],
               'enemy3health': self.our3.health,
               'enemy3energy': self.our3.energy,
               'sk1exist': sk1detarg[0],
               'sk1posx': sk1detarg[1],
               'sk1posy': sk1detarg[2],
               'sk1speedx': sk1detarg[3],
               'sk1speedy': sk1detarg[4],
               'sk2exist': sk2detarg[0],
               'sk2posx': sk2detarg[1],
               'sk2posy': sk2detarg[2],
               'sk2speedx': sk2detarg[3],
               'sk2speedy': sk2detarg[4],
               'sk3exist': sk3detarg[0],
               'sk3posx': sk3detarg[1],
               'sk3posy': sk3detarg[2],
               'sk3speedx': sk3detarg[3],
               'sk3speedy': sk3detarg[4],
               'sk4exist': sk4detarg[0],
               'sk4posx': sk4detarg[1],
               'sk4posy': sk4detarg[2],
               'sk4speedx': sk4detarg[3],
               'sk4speedy': sk4detarg[4]}
        return obs

    def reset(self):
        self.ourgroup.empty()
        self.enemygroup.empty()
        self.sk1group.empty()
        self.sk2group.empty()
        self.sk3group.empty()
        self.sk4group.empty()
        self.our1.reset([random.random() * (self.width - 50), random.random() * (self.height - 50)])
        self.our2.reset([random.random() * (self.width - 50), random.random() * (self.height - 50)])
        self.our3.reset([random.random() * (self.width - 50), random.random() * (self.height - 50)])
        self.ourgroup.add(self.our1)
        self.ourgroup.add(self.our2)
        self.ourgroup.add(self.our3)
        self.enemy1.reset([random.random() * (self.width - 50), random.random() * (self.height - 50)])
        self.enemy2.reset([random.random() * (self.width - 50), random.random() * (self.height - 50)])
        self.enemy3.reset([random.random() * (self.width - 50), random.random() * (self.height - 50)])
        self.enemygroup.add(self.enemy1)
        self.enemygroup.add(self.enemy2)
        self.enemygroup.add(self.enemy3)


