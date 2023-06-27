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

import math
import numpy as np
import numpy.linalg as lalg
import robotic_arm as rba

from math import atan2, sqrt
from numpy import cross, dot
from numpy.linalg import norm

class humanoid_robot(rba.robot):
    a: float
    d: float

    def __init__(self, a3, d5):
        super().__init__([
            rba.rot_link(math.pi / 2,    0,      0, np.pi/2),
            rba.rot_link(math.pi / 2,    0,      0, np.pi/2),
            rba.rot_link(math.pi / 2,    -a3,    0, np.pi/2),
            rba.rot_link(0,              0,      0, np.pi/2),
            rba.rot_link(0,              d5,     0, 0)], 
            Tworkspace = rba.t_rotx(math.pi / 2)
        )
        self.a = a3
        self.d = d5

    def inverse_kinematic(self, P: np.array, **kwargs):
        P0 = np.matmul(
            lalg.inv(self.Tworkspace), 
            np.array([P[0], P[1], P[2], 1]))[:3]
        beta = kwargs['beta'] if 'beta' in kwargs else 0

        p = norm(P0, 2)
        uy = np.array([0, 1, 0])
        uk = cross(P0, cross(P0, uy))
        ap = ((self.a ** 2) + (p ** 2) - (self.d ** 2)) / (2 * p)

        assert self.a ** 2 >= ap ** 2, 'ap^2 ({0}) > a^2 ({0})'.format(ap ** 2, self.a ** 2)

        at = sqrt((self.a ** 2) - ap * ap)
        Ap =  ap * rba.unitary_vector(P0)
        At =  at * rba.unitary_vector(rba.rotate(uk, P0, beta))
        A = Ap + At
        D = P0 - A

        q1 = atan2(A[0], -A[1])
        q2 = atan2(-A[2], norm(A[:2], 2))
        q4 = atan2(
            norm(cross(A, D), 2) / (self.a * self.d), 
            dot(A, D) / (self.a * self.d))

        sq1, sq2, sq4 = np.sin(np.array([q1, q2, q4]))
        cq1, cq2, cq4 = np.cos(np.array([q1, q2, q4]))

        ax = np.array([
            [- self.d * sq1 * sq2 * sq4, self.d * cq1 * sq4],
            [self.d * cq1 * sq2 * sq4, self.d * sq1 * sq4] ])
        b = np.array([
            P0[0] - (self.a * sq1 * cq2 + self.d * sq1 * cq2 * cq4),
            P0[1] + (self.a * cq1 * cq2 + self.d * cq1 * cq2 * cq4) ])

        sq3, cq3 = lalg.solve(ax, b) if lalg.det(ax) != 0 else [0, 1]
        q3 = atan2(sq3, cq3)

        return [q1, q2, q3, q4, 0]



