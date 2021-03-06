#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
import parl
from parl import layers


class AIGameModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 400
        hid2_size = 300
        hid3_size = 200
        hid4_size = 100

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=hid3_size, act='relu')
        self.fc4 = layers.fc(size=hid4_size, act='relu')
        self.fc5 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        hid4 = self.fc4(hid3)
        means = self.fc5(hid4)
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid1_size = 400
        hid2_size = 300
        hid3_size = 200
        hid4_size = 100

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=hid3_size, act='relu')
        self.fc4 = layers.fc(size=hid4_size, act='relu')
        self.fc5 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat = layers.concat([hid1, act], axis=1)
        hid2 = self.fc2(concat)
        hid3 = self.fc3(hid2)
        hid4 = self.fc4(hid3)
        Q = self.fc5(hid4)
        Q = layers.squeeze(Q, axes=[1])
        return Q