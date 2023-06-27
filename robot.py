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

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from math import pi, sin, cos
from matplotlib import animation
from humanoid_arm import humanoid_robot

robot = humanoid_robot(a3 = 50, d5 = 50)

def animate(ax, robot: humanoid_robot, q, i):
    ax.clear()
    
    robot.draw_config(ax, q, i)
    robot.draw_path(ax, q, i)

    handles = [mpatches.Patch(
        label='Q{0} = {1}'.format(j + 1, round(q[i][j], 2))) 
        for j in range(0, len(q[i]))]

    ax.set_xlim([-50, 75])
    ax.set_ylim([-50, 75])
    ax.set_zlim([-50, 75])
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.legend(handles = handles)

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

B = 20
pX = [60 for i in range(0, 100)]
pY = [
    (-B * i / 33 ) if 0 <= i and i < 33 else \
    (-B + 2 * B * (i - 33) / 33) if 33 <= i < 66 else \
    (B - B * (i - 66) / 33)
    for i in range(0, 100)]
pZ = [
    (B - 2 * B * i / 33) if 0<= i and i < 33 else \
    -B if 33<= i and i < 66 else \
    (-B + 2 * B * (i - 66) / 33)
    for i in range(0, 100)
]

pOfQ = [np.array([pX[i], pY[i], pZ[i]]) for i in range(0, 100)]
qOfP = [robot.inverse_kinematic(pOfQ[i], beta = -pi/6) for i in range(0, 100)]

animateFunction = lambda i, *fargs: animate(ax, robot, qOfP, i)
robotAnimation = animation.FuncAnimation(fig, animateFunction, interval=50, frames=100)

plt.show()
