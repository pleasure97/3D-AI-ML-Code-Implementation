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


def save_object_mesh(mesh_path, vertices, faces):
    file = open(mesh_path, 'w')
    for vertex in vertices:
        file.write('v {:.4f} {:.4f} {:.4f}\n'.format(vertex[0], vertex[1], vertex[2]))
    for face in faces:
        face_plus = face + 1
        file.write('f {} {} {}\n'.format(face_plus[0], face_plus[1], face_plus[2]))
    file.close()


def read_material_file(file_name):
    materials = {}
    with open(file_name) as f:
        lines = f.read().splitlines()

    for line in lines:
        if line:
            split_line = line.strip().split(' ', 1)
            if len(split_line) < 2:
                continue

            prefix, data = split_line[0], split_line[1]
            if 'newmtl' in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if data:
                    split_data = data.strip().split(' ')
                    if 'map' in prefix:
                        material[prefix] = split_data[-1].split('\\')[-1]
                    elif len(split_data) > 1:
                        material[prefix] = tuple(float(d) for d in split_data)
                    else:
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)

    return materials


def load_object_mesh_material(mesh_file):
    vertex_data = []
    normal_data = []
    uv_data = []

    face_data = []
    face_normal_data = []
    face_uv_data = []

    # face per material
    face_data_material = []
    face_normal_data_material = []
    face_uv_data_material = []

    # current material name
    material_data = None
    current_material = None

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            normal_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'mtllib':
            material_data = read_material_file(mesh_file.replace(mesh_file.split('/')[-1], values[1]))
        elif values[0] == 'usemtl':
            current_material = values[1]
        elif values[0] == 'f':
            # local triangle data
            local_face_data = []
            local_face_uv_data = []
            local_face_normal_data = []

            def split_value(value, idx=0):
                func = lambda x: int(x.split('/')[idx] if int(x.split('/')[idx]) < 0 else int(x.split('/')[idx]) - 1)
                return list(map(func, value))

            # quad mesh
            if len(values) > 4:
                f = split_value(values[1:4], idx=0)
                local_face_data.append(f)
                f = split_value([values[3], values[4], values[1]], idx=0)
                local_face_data.append(f)
            # triangle mesh
            else:
                f = split_value(values[1:4], idx=0)
                local_face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = split_value(values[1:4], idx=1)
                    local_face_uv_data.append(f)
                    f = split_value([values[3], values[4], values[1]], idx=1)
                    local_face_uv_data.append(f)
                # triangle mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = split_value(values[1:4], idx=1)
                    local_face_uv_data.append(f)

            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = split_value(values[1:4], idx=2)
                    local_face_normal_data.append(f)
                    f = split_value([values[3], values[4], values[1]], idx=2)
                    local_face_normal_data.append(f)
                # triangle mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = split_value(values[1:4], idx=2)
                    local_face_normal_data.append(f)

            face_data += local_face_data
            face_uv_data += local_face_uv_data
            face_normal_data += local_face_normal_data

            if current_material is not None:
                if current_material not in face_data_material.keys():
                    face_data_material[current_material] = []
                if current_material not in face_uv_data_material.keys():
                    face_uv_data_material[current_material] = []
                if current_material  not in face_normal_data_material.keys():
                    face_normal_data_material[current_material] = []
                face_data_material[current_material] += local_face_data
                face_uv_data_material[current_material] += local_face_uv_data
                face_normal_data_material[current_material] += local_face_normal_data

    vertices = np.array(vertex_data)
    faces = np.array(face_data)

    normals = np.array(normal_data)
    normals = normalize_v3(normals)
    face_normals = np.array(face_normal_data)

    uvs = np.array(uv_data)
    face_uvs = np.array(face_uv_data)

    out_tuple = (vertices, faces, normals, face_normals, uvs, face_uvs)

    if current_material is not None and material_data is not None:
        for key in face_data_material:
            face_data_material[key] = np.array(face_data_material[key])
            face_uv_data_material[key] = np.array(face_uv_data_material[key])
            face_normal_data_material[key] = np.array(face_normal_data_material[key])

        out_tuple += (face_data_material, face_normal_data_material, face_uv_data_material, material_data)

    return out_tuple

def load_object_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    normal_data = []
    uv_data = []

    face_data = []
    face_normal_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, 'r')
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            normal_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'f':
            def _split_value(value, idx=0):
                return list(map(lambda x : int(x.split('/')[idx]), value))
            # quad mesh
            if len(values) > 4:
                f = _split_value(values[1:4], idx=0)
                face_data.append(f)
                f = _split_value([values[3], values[4], values[1]], idx=0)
                face_data.append(f)
            # triangle mesh
            else:
                f = _split_value(values[1:4], idx=0)
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = _split_value(values[1:4], idx=1)
                    face_uv_data.append(f)
                    f = _split_value([values[3], values[4], values[1]], idx=1)
                    face_uv_data.append(f)
                # triangle mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = _split_value(values[1:4], idx=1)
                    face_uv_data.append(f)

            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = _split_value(values[1:4], idx=2)
                    face_normal_data.append(f)
                    f = _split_value([values[3], values[4], values[1]], idx=2)
                    face_normal_data.append(f)
                # triangle mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = _split_value(values[1:4], idx=2)
                    face_normal_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        normals = np.array(normal_data)
        if normals.shape[0] == 0:
            normals = compute_normal(vertices, faces)
            face_normals = faces
        else:
            normals = normalize_v3(normals)
            face_normals = np.array(face_normal_data) - 1
        return vertices, faces, normals, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        normals = np.array(normal_data)
        normals = normalize_v3(normals)
        face_normals = np.array(face_normal_data) - 1
        return vertices, faces, normals, face_normals

    return vertices, faces

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape = (n,3)'''
    norms = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 1e-8
    norms[norms < eps] = eps
    arr[:, 0] /= norms
    arr[:, 1] /= norms
    arr[:, 2] /= norms
    return arr

def compute_normal(vertices, faces):
    # Create a zero-filled array with the same type and shape as our vertices i.e., per veretex normal
    normal = np.zeros(vertices.shape, dtype=vertices.dtype)

    # Create an indexed view into the vertex array using the array of 3 indices for triangles
    triangles = vertices[faces]

    # Calculate the normal for all the triangles, by taking the cross product of the vector v1-v0, and v2-v0 in each
    n = np.cross(triangles[::, 1] - triangles[::, 0], triangles[::, 2] - triangles[::, 0])

    # n is now an array of normals per triangle. The length of each normal is dependent with the vertices.
    # we need to normalize these so that our next step weights each normal equality
    normalize_v3(n)

    normal[faces[:, 0]] += n
    normal[faces[:, 1]] += n
    normal[faces[:, 2]] += n

    normalize_v3(normal)

    return normal


def compute_tangent(vertices, faces, normals, uvs, face_uvs):
    # compute tangent and bitangent
    c1 = np.cross(normals, np.array([0., 1., 0.]))
    tangent = c1
    normalize_v3(tangent)
    bitangent = np.cross(normals, tangent)

    return tangent, bitangent

if __name__ == '__main__':
    points, triangle, normal, triangle_normal, uvs, triangle_uvs \
        = load_object_mesh('/home/ICT2000/ssaito/Documents/Body/tmp/Baseball_Pitching/0012.obj', True, True)
    compute_tangent(points, triangle, uvs, triangle_uvs)