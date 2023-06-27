# Copyright 2023 Jerson Leonardo Huerfano Romero
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

from abc import ABC, abstractclassmethod
from typing import Callable
import numpy as np
import numpy.linalg as linalg
import math
import quaternions as quat

def t_rotx(q):
    return np.array([ \
        [1, 0          , 0           , 0], \
        [0, math.cos(q), -math.sin(q), 0], \
        [0, math.sin(q), math.cos(q) , 0], \
        [0, 0          , 0           , 1]]).round(decimals = 5)

def t_rotz(q):
    return np.array([ \
        [math.cos(q), -math.sin(q), 0, 0], \
        [math.sin(q), math.cos(q) , 0, 0], \
        [0          , 0           , 1, 0], \
        [0          , 0           , 0, 1]]).round(decimals = 5)

def t_movex(q):
    return np.array([ \
        [1, 0, 0, q], \
        [0, 1, 0, 0], \
        [0, 0, 1, 0], \
        [0, 0, 0, 1]]).round(decimals = 5)

def t_movez(q):
    return np.array([ \
        [1, 0, 0, 0], \
        [0, 1, 0, 0], \
        [0, 0, 1, q], \
        [0, 0, 0, 1]]).round(decimals = 5)

def dh_transform(theta, d, a, alpha):
    return np.matmul(np.matmul(t_rotz(theta), t_movez(d)), np.matmul(t_rotx(alpha), t_movex(a)))

def rot_link(theta, d, a, alpha):
    return lambda q: dh_transform(q + theta, d, a, alpha)

def prism_link(theta, d, a, alpha):
    return lambda q: dh_transform(theta, d + q, a, alpha)

def unitary_vector(r):
    return r / linalg.norm(r, 2)

def rotate(v: np.array, axis: np.array, angle):
    p_v = quat.quaternion(0, v)
    q = quat.quaternion(math.cos(angle / 2), math.sin(angle / 2) * unitary_vector(axis))
    _, p_vr = quat.q_product(quat.q_product(q, p_v), quat.q_conjugate(q)).parts()
    return p_vr

class robot(ABC):
    T: list[Callable[[int], np.array]]
    Tworkspace: np.array

    def __init__(self, T: list[Callable[[int], np.array]], Tworkspace = np.eye(4)):
        self.T = T
        self.Tworkspace = Tworkspace

    def state_of(self, q:list[float], k: int = None):
        Tresult = self.Tworkspace

        n = len(self.T)
        k = n if k is None else min(k, n)

        for i in range(0, k):
            Ti = self.T[i]
            Tresult = np.matmul(Tresult, Ti(q[i])) 
        
        return Tresult
    
    def position_of(self, q:list[float], k: int = None):
        return np.matmul(self.state_of(q, k), np.array([0, 0, 0, 1]))
    
    def orientation_of(self, q:list[float], k: int = None):
        return self.state_of(q, k)[:3,:3]

    @abstractclassmethod
    def inverse_kinematic(P: np.array, **kwargs):
        pass

    def draw_config(self, ax, q: list[np.array], k = None):
        n = len(self.T)

        origin = [0, 0, 0, 1]
        confX = []
        confY = []
        confZ = []

        for i in range(0, n + 1):
            qk = q if k is None else q[k]
            o_i =  np.matmul(self.state_of(qk, i), origin)

            confX.append(o_i[0])
            confY.append(o_i[1])
            confZ.append(o_i[2])
            
            ax.scatter(o_i[0], o_i[1], o_i[2])

        ax.plot(confX, confY, confZ, label = 'Config {0}'.format('' if k is None else k))
        ax.scatter(o_i[0],o_i[1],o_i[2], label = 'Point {0}'.format(o_i[:3].round(decimals = 2)))

    def draw_path(self, ax, q: list[np.array], k):
        origin = [0, 0, 0, 1]
        
        path = [np.matmul(self.state_of(q[i]), origin) for i in range(0, k)]
        pathx = [path[i - 1][0] for i in range(1, k)]
        pathy = [path[i - 1][1] for i in range(1, k)]
        pathz = [path[i - 1][2] for i in range(1, k)]
        
        ax.plot(pathx, pathy, pathz, linestyle = 'dashed', label = 'Path')
