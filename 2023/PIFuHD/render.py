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
from OpenGL.GL import *
from OpenGL.GLUT import *
from framework import *

_glut_window = None


class Render:
    def __init__(self,
                 width=1600.,
                 height=1200.,
                 name='GL Renderer',
                 program_files=['simple.fs', 'simple.vs'],
                 color_size=1.,
                 ms_rate=1.):
        self.width = width
        self.height = height
        self.name = name
        ''' 
        GLUT_DOUBLE - double buffering / render in one buffer and display on screen in another
        GLUT_DOUBLE allows the screen to be rendered flicker-free. 
        GLUT_RGB - using RGB color model 
        GLUT_DEPTH - using depth buffering / take depth information into account..
        GLUT_DEPTH allows you to determine the visibility and rendering order of objects based on their depth info.
        '''
        self.display_mode = GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH
        self.use_inverse_depth = False

        global _glut_window

        if _glut_window is None:
            # Must call glutInit() to use GLUT
            glutInit()
            # Call glutInitDisplayMethod() to set display options such as rendering context, color model, buffering.
            glutInitDisplayMode(self.display_mode)
            # Set initial size and position of the window by using glutInitWindowSize() and glutInitWindowPosition().
            glutInitWindowSize(self.width, self.height)
            glutInitWindowPosition(0, 0)
            # Create new window using glutCreateWindow and the name of the window is 'My Render.'.
            _glut_window = glutCreateWindow("My Render.")
            # Activate depth buffering to prevent 3D objects from mixing.
            glEnable(GL_DEPTH_TEST)

            # Clamp color coordinates
            glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE)

            # init program
            shader_list = []

            for program_file in program_files:
                _, ext = os.path.splitext(program_file)
                if ext == '.vs':
                    shader_list.append(load_shader(GL_VERTEX_SHADER, program_file))
                elif ext == '.gs':
                    shader_list.append(load_shader(GL_GEOMETRY_SHADER, program_file))
                elif ext == '.fs':
                    shader_list.append(load_shader(GL_FRAGMENT_SHADER, program_file))

            self.program = create_program(shader_list)

            for shader in shader_list:
                glDeleteShader(shader)

            # glGetUniformLocation retrieves the location of the uniform variable defined in the program.
            # ModelMat means model matrix that specifies location, rotation, and size of 3D model.
            # PerspMat means perspective matrix that is a kind of projection matrix.
            # perspective matrix is made by using the camera's field of view, aspect ratio, and values of near plane
            # and far plane.
            self.model_mat_uniform = glGetUniformLocation(self.program, 'ModelMat')
            self.persp_mat_uniform = glGetUniformLocation(self.program, 'PerspMat')

            # generate a buffer object and parameter 1 means generate one buffer.
            self.vertex_buffer = glGenBuffers(1)

            # Init screen quad program and buffer
            self.quad_program, self.quad_buffer = self.init_quad_program()

    def init_quad_program(self):
        shader_list = []

        shader_list.append(load_shader(GL_VERTEX_SHADER, "quad.vs"))
        shader_list.append(load_shader(GL_FRAGMENT_SHADER, "quad.fs"))

        quad_program = create_program(shader_list)

        for shader in shader_list:
            glDeleteShader(shader)

        # vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        # composed of positions and texCoords
        quad_vertices = np.array(
            [-1., 1., 0., 1.,
             -1., -1., 0., 0.,
             1., -1., 1., 0.,

             -1., 1., 0., 1.,
             1., -1., 1., 0.,
             1., 1., 1., 1.]
        )

        quad_buffer = glGenBuffers(1)
        # bind a buffer ID to GL_ARRAY_BUFFER type buffer
        # OpenGL is designed to be a state machine that keeps track of which objects are being used.
        # OpenGL automatically updates all state associated with related object and frees the previously used object
        # so that another object can be used
        glBindBuffer(GL_ARRAY_BUFFER, quad_buffer)
        #
        glBufferData(GL_ARRAY_BUFFER, quad_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        return quad_program, quad_buffer


