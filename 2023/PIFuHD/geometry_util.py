'''
MIT License
Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np


def identity():
    return np.identity(4, dtype=np.float32)


def orthogonal(left, right, bottom, top, z_near, z_far):

    result = identity()

    result[0][0] = 2 / (right - left)
    result[1][1] = 2 / (top - bottom)
    result[2][2] = - 2 / (z_far - z_near)
    result[3][0] = - (right + left) / (right - left)
    result[3][1] = - (top + bottom) / (top - bottom)
    result[3][2] = - (z_far + z_near) / (z_far - z_near)

    return result.T

