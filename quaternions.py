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

import numpy as np

class quaternion:
    real_part: float
    vector_part: np.array

    def __init__(self, real_part: float, vector_part):
        self.real_part = real_part
        self.vector_part = np.array(vector_part)
    
    def parts(self):
        return self.real_part, self.vector_part


def q_conjugate(q: quaternion):
    return quaternion(q.real_part, -1 * q.vector_part)

def q_product(qa: quaternion, qb: quaternion):
    return quaternion(
        qa.real_part * qb.real_part - np.dot(qa.vector_part, qb.vector_part)
        , qa.real_part * qb.vector_part + qb.real_part * qa.vector_part
            + np.cross(qa.vector_part, qb.vector_part))