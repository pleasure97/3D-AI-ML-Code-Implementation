from OpenGL.GL import *
import os


# Define function that creates and compiles shaders according to the given type (a GL enum value) and a shader program
# that contains a GLSL program.
def load_shader(shader_type, shader_file):
    # check if file exists, get full path name
    str_file_name = find_file_or_throw(shader_file)
    shader_data = None
    with open(str_file_name, 'r') as f:
        shader_data = f.read()

    # create a shader that determines color, illumination, texture, and other features related to model's vertex data.
    shader = glCreateShader(shader_type)
    # set the source code of the shader, and the source code is passed as a string.
    # glShaderSource attaches the source code to the specified shader object.
    glShaderSource(shader, shader_data)
    # compile the shader object and check if an error exists during compiling.
    glCompileShader(shader)

    # glGetShaderiv checks the compilation state of the compiled shaders.
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status == GL_FALSE:
        str_info_log = glGetShaderInfoLog(shader)
        str_shader_type = ""
        # GL_VERTEX_SHADER is used to determine the location, color, texture, and other features of vertices.
        # GL_VERTEX_SHADER loads and transforms vertex data, and passes transformed vertex data to pixel shader.
        if shader_type is GL_VERTEX_SHADER:
            str_shader_type = "vertex"
        # GL_GEOMETRY SHADER takes vertex data and then create new vertices, delete or modify existing vertices.
        # GL_GEOMETRY SHADER implements geometrical algorithms to process vertex data.
        elif shader_type is GL_GEOMETRY_SHADER:
            str_shader_type = "geometry"
        # GL_FRAGMENT_SHADER is used to determine a color by pixel.
        # GL_FRAGMENT_SHADER implements texture mapping, illumination, transparency, shadows and other effects.
        elif shader_type is GL_FRAGMENT_SHADER:
            str_shader_type = "fragment"

        print("Compilation failure for " + str_shader_type + " shader:\n" + str(str_info_log))

    return shader


# Define helper function to locate and open the target file (passed in as a string.).
# Return the full path to the file as a string.
def find_file_or_throw(str_base_name):
    if os.path.isfile(str_base_name):
        return str_base_name

    LOCAL_FILE_DIR = 'data' + os.sep
    GLOBAL_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'data' + os.sep

    str_file_name = LOCAL_FILE_DIR + str_base_name
    if os.path.isfile(str_file_name):
        return str_file_name

    str_file_name = GLOBAL_FILE_DIR + str_base_name
    if os.path.isfile(str_file_name):
        return str_file_name

    raise IOError('Could not find target file ' + str_base_name)


# Define function that accepts a list of shaders, compiles them, and returns a handle to the compiled program.
def create_program(shader_list):
    # create a program so that it will be able to attach the compiled shader.
    program = glCreateProgram()

    for shader in shader_list:
        # attach the compiled shader to the program object
        glAttachShader(program, shader)

    # Link program objects to create the final program.
    glLinkProgram(program)

    # get the program information to check if program is linked
    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status == GL_FALSE:
        # get the program log information
        str_info_log = glGetProgramInfoLog(program)
        print("Linker failure :\n" + str(str_info_log))

    for shader in shader_list:
        glDetachShader(program, shader)

    return program
