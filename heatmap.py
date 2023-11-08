# python heatmap.py --opencv --bop --overlay --path_json_gt /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/bop_2023_hope_paper/hope/val/000001/scene_gt.json --path_json_gu /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/bop_2023_hope_paper/hope/val/000001/scene_error_deg_040_trans_016.json --objs_folder /media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/bop_2023_hope_paper/hope/models/ --contour --spp 100 
import nvisii as visii 
import argparse
import numpy as np

import simplejson as json
import pyrender 
import cv2 
import numpy as np
import os 
import glob
import pyrr 
from scipy import spatial

import trimesh
import matplotlib.pyplot as plt
import xatlas
import time



parser = argparse.ArgumentParser()

parser.add_argument(
    '--spp', 
    default=100,
    type=int,
    help = "number of sample per pixel, higher the more costly"
)

parser.add_argument(
    '--bop',
    action='store_true',
    default=False,
    help = "Use the bop format for the camera poses and .ply format \
            for the 3d object."
)
parser.add_argument(
    '--opencv',
    action='store_true',
    default=False,
    help = "Use the bop format for the camera poses and .ply format \
            for the 3d object."
)
parser.add_argument(
    '--bop_frame',
    default=0,
    type=int,
    help = "Which scene to load, only applied to bop format."
)

parser.add_argument(
    '--contour',
    action='store_true',
    default=False,
    help = "Only draw the contour instead of the 3d model overlay"
)

parser.add_argument(
    '--overlay',
    action='store_true',
    default=False,
    help = "add the overlay"
)

parser.add_argument(
    '--gray',
    action='store_true',
    default=False,
    help = "draw the 3d model in gray"
)

parser.add_argument(
    '--path_json_gt',
    required=True,
    help = "path to the json files you want loaded,\
            it assumes that there is a png accompanying."
)

parser.add_argument(
    '--path_json_gu',
    required=True,
    help = "path to the json files you want loaded,\
            it assumes that there is a png accompanying."
)

parser.add_argument(
    '--objs_folder',
    default='content/',
    help = "object to load folder, should follow YCB structure"
)

parser.add_argument(
    '--scale',
    default=1,
    type=float,
    help='Specify the scale of the target object(s). If the obj mesh is in '
         'meters -> scale=1; if it is in cm -> scale=0.01.'
)
parser.add_argument(
    '--adds',
    action='store_true',
)
parser.add_argument(
    '--raw',
    action='store_true',
    help='make the displayed textured raw, so you have a non smooth texture, but at least it is a little faster.'
)

parser.add_argument(
    '--out',
    default='heatmap.png',
)

parser.add_argument(
    '--max_distance',
    type = float,
    default=0.1,
)
parser.add_argument(
    '--outf_bop',
    type=str,
    default='value.csv'
)
parser.add_argument(
    '--scene_id',
    default=None,
)
parser.add_argument(
    '--keep',
    action='store_true',
    help='do not apply the multiplcation factor for megapose'
)
opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #

def create_obj(
    name = 'name',
    path_obj = "",
    path_tex = None,
    scale = 1, 
    rot_base = None, #visii quat
    pos_base = (-10,-10,-10), # visii vec3
    ):

    
    # This is for YCB like dataset
    if path_obj in create_obj.meshes:
        obj_mesh = create_obj.meshes[path_obj]
    else:
        # print(path_obj)

        obj_mesh = visii.mesh.create_from_obj(name, path_obj)
        create_obj.meshes[path_obj] = obj_mesh

    
    obj_entity = visii.entity.create(
        name = name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(1) # default is 1  

    if not path_tex is None and os.path.exists(path_tex):

        if path_tex in create_obj.textures:
            obj_texture = create_obj.textures[path_tex]
        else:
            obj_texture = visii.texture.create_from_image(name,path_tex)
            create_obj.textures[path_tex] = obj_texture


        obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    if not rot_base is None:
        obj_entity.get_transform().set_rotation(rot_base)
    if not pos_base is None:
        obj_entity.get_transform().set_position(pos_base)
    print(f' created: {obj_entity.get_name()}')
    return obj_entity

create_obj.meshes = {}
create_obj.textures = {}

# # # # # # # # # # # # # # # # # # # # # # # # #


visii.initialize_headless()
# visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create(
        name = "camera", 
    )
)

camera.get_transform().look_at(
    visii.vec3(0,0,-1), # look at (world coordinate)
    visii.vec3(0,1,0), # up vector
    visii.vec3(0,0,0), # camera_origin    
)

visii.set_camera_entity(camera)

visii.set_dome_light_intensity(1)

try:
    visii.set_dome_light_color(visii.vec3(1, 1, 1), 0)
except TypeError:
    # Support for alpha transparent backgrounds was added in nvisii ef1880aa,
    # but as of 2022-11-03, the latest released version (1.1) does not include
    # that change yet.
    print("WARNING! Your version of NVISII does not support alpha transparent backgrounds yet; --contour will not work properly.")
    visii.set_dome_light_color(visii.vec3(1, 1, 1))

# # # # # # # # # # # # # # # # # # # # # # # # #

# LOAD THE SCENE 

objects_added = []
dope_trans = []
gt = None

with open(opt.path_json_gt) as f:
    data_json = json.load(f)

with open(opt.path_json_gu) as f:
    data_json_gu = json.load(f)


if opt.bop: 
    # opt.opencv = True
    camera_path = "/".join(opt.path_json_gt.split("/")[:-1]) + "/scene_camera.json"
    with open(camera_path) as f:
        camera_data_json = json.load(f)
        # a little bit of a hack, use the first camera, would need 
    # print(camera_data_json[str(opt.bop_frame)]["cam_K"])
    # raise()
    keys = list(camera_data_json.keys())
    if not str(opt.bop_frame) in keys:
        opt.bop_frame = keys[0]

    data_json['camera_data'] = {}
    data_json['camera_data']['intrinsics'] = {}
    data_json['camera_data']['intrinsics']['fx'] = camera_data_json[str(opt.bop_frame)]["cam_K"][0]
    data_json['camera_data']['intrinsics']['fy'] = camera_data_json[str(opt.bop_frame)]["cam_K"][4]
    data_json['camera_data']['intrinsics']['cx'] = camera_data_json[str(opt.bop_frame)]["cam_K"][2]
    data_json['camera_data']['intrinsics']['cy'] = camera_data_json[str(opt.bop_frame)]["cam_K"][5]

    # load an image for resolution 

    path_imgs = "/".join(opt.path_json_gt.split("/")[:-1]+ ['rgb/'])
    path_imgs = sorted(glob.glob(path_imgs + "*.png"))
    opt.im_path = path_imgs[0]
    img = cv2.imread(path_imgs[0])
    data_json['camera_data']['height'] = img.shape[0]
    data_json['camera_data']['width'] = img.shape[1]

# set the camera
if "camera_data" in data_json.keys() and "intrinsics" in data_json['camera_data'].keys():
    intrinsics = data_json['camera_data']['intrinsics']
    im_height = data_json['camera_data']['height']
    im_width = data_json['camera_data']['width']

    cam = pyrender.IntrinsicsCamera(intrinsics['fx'],intrinsics['fy'],intrinsics['cx'],intrinsics['cy'])

    proj_matrix = cam.get_projection_matrix(im_width, im_height)

    proj_matrix = visii.mat4(
            proj_matrix.flatten()[0],
            proj_matrix.flatten()[1],
            proj_matrix.flatten()[2],
            proj_matrix.flatten()[3],
            proj_matrix.flatten()[4],
            proj_matrix.flatten()[5],
            proj_matrix.flatten()[6],
            proj_matrix.flatten()[7],
            proj_matrix.flatten()[8],
            proj_matrix.flatten()[9],
            proj_matrix.flatten()[10],
            proj_matrix.flatten()[11],
            proj_matrix.flatten()[12],
            proj_matrix.flatten()[13],
            proj_matrix.flatten()[14],
            proj_matrix.flatten()[15],
    )
    proj_matrix = visii.transpose(proj_matrix)

    camera.get_camera().set_projection(proj_matrix)
else:
    im_height = 512
    im_width = 512
    intrinsics = {  "cx": 964.957,
                    "cy": 522.586,
                    "fx": 1390.53,
                    "fy": 1386.99,
                }

    cam = pyrender.IntrinsicsCamera(intrinsics['fx'],intrinsics['fy'],intrinsics['cx'],intrinsics['cy'])

    proj_matrix = cam.get_projection_matrix(im_width, im_height)
    # print(proj_matrix)
    proj_matrix = visii.mat4(
            proj_matrix.flatten()[0],
            proj_matrix.flatten()[1],
            proj_matrix.flatten()[2],
            proj_matrix.flatten()[3],
            proj_matrix.flatten()[4],
            proj_matrix.flatten()[5],
            proj_matrix.flatten()[6],
            proj_matrix.flatten()[7],
            proj_matrix.flatten()[8],
            proj_matrix.flatten()[9],
            proj_matrix.flatten()[10],
            proj_matrix.flatten()[11],
            proj_matrix.flatten()[12],
            proj_matrix.flatten()[13],
            proj_matrix.flatten()[14],
            proj_matrix.flatten()[15],
    )
    proj_matrix = visii.transpose(proj_matrix)
    # print(proj_matrix)
    camera.get_camera().set_projection(proj_matrix)

if opt.bop: 
    # get the objects to load. 
    scene_objs = data_json[str(opt.bop_frame)]
    data_json['objects'] = []
    for iobj, obj in enumerate(scene_objs):
        # print(obj)
        to_add = {}
        to_add['class'] = str(obj['obj_id'])+'_gt'+str(iobj)
        to_add['location'] = obj['cam_t_m2c']

        # figuring out the rotation now
        rot = obj['cam_R_m2c']
        m = pyrr.Matrix33(
            [
                [rot[0],rot[1],rot[2]],
                [rot[3],rot[4],rot[5]],
                [rot[6],rot[7],rot[8]],
                # [rot[0],rot[3],rot[6]],
                # [rot[1],rot[4],rot[7]],
                # [rot[2],rot[5],rot[8]],                
            ])
        quat = m.quaternion
        to_add['quaternion_xyzw'] = [quat.x,quat.y,quat.z,quat.w]
        data_json['objects'].append(to_add)

    for iobj,obj in enumerate(data_json_gu[str(opt.bop_frame)]):
        # print(obj)
        to_add = {}
        to_add['class'] = str(obj['obj_id'])+'_gu'+str(iobj)
        # to_add['location'] = obj['cam_t_m2c']

        if not opt.keep and "mega" in opt.path_json_gu:
            to_add['location'] = [
                obj['cam_t_m2c'][0]*1000,
                obj['cam_t_m2c'][1]*1000,
                obj['cam_t_m2c'][2]*1000
            ]
            # print(to_add['location'])
        else:
            to_add['location'] = obj['cam_t_m2c']

        # figuring out the rotation now
        rot = obj['cam_R_m2c']
        if len(rot) == 3:
            rot = np.array(rot).flatten()
        m = pyrr.Matrix33(
            [
                [rot[0],rot[1],rot[2]],
                [rot[3],rot[4],rot[5]],
                [rot[6],rot[7],rot[8]],
                # [rot[0],rot[3],rot[6]],
                # [rot[1],rot[4],rot[7]],
                # [rot[2],rot[5],rot[8]],                
            ])
        quat = m.quaternion
        to_add['quaternion_xyzw'] = [quat.x,quat.y,quat.z,quat.w]
        data_json['objects'].append(to_add)

for i_obj, obj in enumerate(data_json['objects']):
    name = obj['class'].split('_')[0]
    # hack because notation is old
    if name == '003':
        name = '003_cracker_box_16k'

    print('loading',name)
    if opt.bop:
        entity_visii = create_obj(
            name = obj['class'],
            path_obj = f"{opt.objs_folder}/obj_{str(name).zfill(6)}.ply",
            path_tex = f"{opt.objs_folder}/obj_{str(name).zfill(6)}.png",
            scale = .01, 
            rot_base = None
        )      

    else:    
        entity_visii = create_obj(
            name = obj['class'] + "_" + str(i_obj),
            path_obj = opt.objs_folder + "/"+name + "/google_16k/textured.obj",
            path_tex = opt.objs_folder + "/"+name + "/google_16k/texture_map_flat.png",
            scale = opt.scale,
            rot_base = None
        )        
    if "gu" in obj['class']:
        entity_visii.get_material().set_base_color(visii.vec3(0,0,1))
    else: 
        entity_visii.get_material().set_base_color(visii.vec3(0,1,0))

    # entity_visii.get_material().set_alpha(0.5)
    entity_visii.get_material().set_roughness(1)
    entity_visii.get_material().set_metallic(0)
    pos = obj['location']
    rot = obj['quaternion_xyzw']

    entity_visii.get_transform().set_rotation(
        visii.quat(
            rot[3],
            rot[0],
            rot[1],
            rot[2],
        )
    )
    if opt.opencv:
        entity_visii.get_transform().set_position(
            visii.vec3(
                pos[0]/100,
                pos[1]/100,
                pos[2]/100,
            )
        )
    elif opt.bop: 
        entity_visii.get_transform().set_position(
            visii.vec3(
                pos[0]/1000,
                pos[1]/1000,
                pos[2]/1000,
            )
        )
    else:
        entity_visii.get_transform().set_position(
            visii.vec3(
                pos[0],
                pos[1],
                pos[2],
            )
        )
    if opt.opencv or opt.bop:
        entity_visii.get_transform().rotate_around(visii.vec3(0,0,0),visii.angleAxis(visii.pi(), visii.vec3(1,0,0)))

# # # # # # # # # # # # # # # # # # # # # # # # #

scene_id = opt.path_json_gt.replace("//",'/')
scene_id = scene_id.split('/')[-2]

visii.render_to_file(
    width=im_width, 
    height=im_height, 
    samples_per_pixel=100,
    file_path=f'{opt.out.replace(".png","_2.png")}'
)

# raise()
# # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd 

for iobj,obj in enumerate(data_json[str(opt.bop_frame)]):
    nv_gt = str(obj['obj_id'])+'_gt'+str(iobj)
    nv_gu = str(obj['obj_id'])+'_gu'+str(iobj)

    visii_gt = visii.entity.get(nv_gt)
    
    visii_gu = visii.entity.get(nv_gu)
    if visii_gu is None:
        print(nv_gt,nv_gu)
    try:
        vertices = np.array(visii_gt.get_mesh().get_vertices())
    except:
        print(f'problem with {nv_gt}')
        # continue
    triangles = np.array(visii_gt.get_mesh().get_triangle_indices()).reshape(-1,3)
    points_gt = []
    points_gu = []
    dist_add_all = []

    ### make a vertex colored mesh
    max_distance =  opt.max_distance
    print(vertices.shape)
    print(triangles.shape)

    vmapping, indices, uvs = xatlas.parametrize(vertices, triangles)
    
    os.makedirs("objects_tmp",exist_ok=True)
    # if not os.path.exists(f"objects_tmp/{obj['obj_id']}.obj"):
    #     xatlas.export(f"objects_tmp/{obj['obj_id']}.obj", vertices[vmapping], indices, uvs)
    os.makedirs("objects_tmp",exist_ok=True)
    xatlas.export(f"objects_tmp/{obj['obj_id']}.obj", vertices[vmapping], indices, uvs)

    mesh = trimesh.load(f"objects_tmp/{obj['obj_id']}.obj")
    vertices = mesh.vertices

    for i in range(len(vertices)):
        v = visii.vec4(vertices[i][0],vertices[i][1],vertices[i][2],1)
        p0 = visii_gt.get_transform().get_local_to_world_matrix() * v
        p1 = visii_gu.get_transform().get_local_to_world_matrix() * v
        points_gt.append([p0[0],p0[1],p0[2]])
        points_gu.append([p1[0],p1[1],p1[2]])
        dist_add_all.append(visii.distance(p0, p1))

    dist_add_all = np.array(dist_add_all)
    dist_adds_all = spatial.distance_matrix(
                    np.array(points_gt), 
                    np.array(points_gu),p=2).min(axis=1)

    dist_add = np.mean(dist_add_all)
    dist_adds = np.mean(dist_adds_all)

    print(dist_add_all.min(),dist_add_all.max())
    max_distance = 0.5
    dist_add_all = np.append(dist_add_all,[0,max_distance])
    dist_add_all[dist_add_all>max_distance] = max_distance

    # COLORING

    # cmap = plt.cm.get_cmap('Reds')
    cmap = plt.cm.get_cmap('turbo')
    # Map the scalar values to colors using the interpolate function
    colors = trimesh.visual.color.interpolate(
        dist_add_all/max_distance, 
        color_map=cmap
    )

    # Set the vertex colors to visualize the heatmap
    c_vis = trimesh.visual.ColorVisuals(mesh,vertex_colors=colors[:-2])

    res = 1024
    texture_image = np.ones((res, res, 3), dtype=np.uint8)*255  # Initialize the texture image (resxres resolution)
    uvs = mesh.visual.uv


    def compute_image(im,triangles,colors):
        # Specify (x,y) triangle vertices
        res = im.shape[0]
        a = triangles[0]
        b = triangles[1]
        c = triangles[2]

        # Specify colors
        red = colors[0]
        green = colors[1]
        blue = colors[2]

        # Make array of vertices
        # ax bx cx
        # ay by cy
        #  1  1  1
        triArr = np.asarray([a[0],b[0],c[0], a[1],b[1],c[1], 1,1,1]).reshape((3, 3))

        # Get bounding box of the triangle
        xleft = min(a[0], b[0], c[0])
        xright = max(a[0], b[0], c[0])
        ytop = min(a[1], b[1], c[1])
        ybottom = max(a[1], b[1], c[1])

        # Build np arrays of coordinates of the bounding box
        xs = range(xleft, xright)
        ys = range(ytop, ybottom)
        xv, yv = np.meshgrid(xs, ys)
        xv = xv.flatten()
        yv = yv.flatten()

        # Compute all least-squares /
        p = np.array([xv, yv, [1] * len(xv)])
        alphas, betas, gammas = np.linalg.lstsq(triArr, p, rcond=-1)[0]

        # Apply mask for pixels within the triangle only
        mask = (alphas >= 0) & (betas >= 0) & (gammas >= 0)
        alphas_m = alphas[mask]
        betas_m = betas[mask]
        gammas_m = gammas[mask]
        xv_m = xv[mask]
        yv_m = yv[mask]

        def mul(a, b) :
            # Multiply two vectors into a matrix
            return np.asmatrix(b).T @ np.asmatrix(a)

        # Compute and assign colors
        colors = mul(red, alphas_m) + mul(green, betas_m) + mul(blue, gammas_m)
        # im[xv_m, yv_m] = colors
        try:
            im[(res-1)-yv_m,xv_m] = colors
        except:
            pass
        # im[res-xv_m,yv_m] = colors

        return im

    def compute_image_wand(im,triangles,colorsIn):
        from wand.image import Image
        from wand.color import Color
        from wand.drawing import Drawing
        from wand.display import display
        # print()
        colors = {
            Color(f"rgb({colorsIn[0][0]},{colorsIn[0][1]},{colorsIn[0][2]})"): triangles[0],
            Color(f"rgb({colorsIn[1][0]},{colorsIn[1][1]},{colorsIn[1][2]})"): triangles[1],
            Color(f"rgb({colorsIn[2][0]},{colorsIn[2][1]},{colorsIn[2][2]})"): triangles[2]
        }

        with Image.from_array(im) as img:
            with img.clone() as mask:
                with Drawing() as draw:
                    points = triangles
                    draw.fill_color = Color('white')
                    draw.polygon(points)
                    draw.draw(mask)
                    img.sparse_color('barycentric', colors)
                    img.composite_channel('all_channels', mask, 'multiply', 0, 0)   
                    # img.format = 'png'
                    # img.save(filename='barycentric_image.png')
                    # display(img)
        return im
    print('paintin start')
    start_time = time.time()
    for triangle in mesh.faces:
        # print(triangle)
        c1,c2,c3 = triangle
        uv1 = (int(uvs[c1][0]*res),int(uvs[c1][1]*res))
        uv2 = (int(uvs[c2][0]*res),int(uvs[c2][1]*res))
        uv3 = (int(uvs[c3][0]*res),int(uvs[c3][1]*res))
        # fill in the colors 
        c_filling = [
            (int(colors[c1][0]),int(colors[c1][1]),int(colors[c1][2])),
            (int(colors[c2][0]),int(colors[c2][1]),int(colors[c2][2])),
            (int(colors[c3][0]),int(colors[c3][1]),int(colors[c3][2]))
        ]


        texture_image = compute_image(texture_image, 
            [uv1,uv2,uv3], c_filling)

        # texture_image = compute_image_wand(texture_image, 
        #     [uv1,uv2,uv3], c_filling)

        # texture_file = cv2.line(texture_image, 
        #     (uv1), 
        #     (uv2), 
        #     (255,255,255), 1)
        # texture_file = cv2.line(texture_image, 
        #     (uv2), 
        #     (uv3), 
        #     (255,255,255), 1)
        # texture_file = cv2.line(texture_image, 
        #     (uv3), 
        #     (uv1), 
        #     (255,255,255), 1)
    print('end painting',time.time()-start_time)


    def find_non_black_in_window(image, x, y, window_size=2):
        half_window = window_size // 2
        window = image[max(0, x - half_window):min(image.shape[0], x + half_window + 1),
                       max(0, y - half_window):min(image.shape[1], y + half_window + 1)]

        # Find the non-black pixels within the window
        non_black_coords = np.where(np.all(window != [0, 0, 0], axis=-1))
        if non_black_coords[0].size > 0:
            index = np.random.randint(non_black_coords[0].size)  # Randomly select a non-black pixel from the window
            non_black_x, non_black_y = non_black_coords[0][index], non_black_coords[1][index]
            return window[non_black_x, non_black_y]
        else:
            None  # Return black pixel if no non-black pixel found

    if opt.raw is False:
        # Convert the texture image to grayscale
        gray_texture = cv2.cvtColor(texture_image, cv2.COLOR_BGR2GRAY)

        # Find the white spots in the grayscale image (you may need to adjust the threshold)
        mask = cv2.inRange(gray_texture, 200, 255)

        # Apply inpainting to fill in the white spots
        texture_image = cv2.inpaint(texture_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


    # black_coords = np.where(grayscale_image >20)

    # for x, y in zip(*black_coords):
    #     c = find_non_black_in_window(texture_image, x, y)
    #     if not c is None:
    #         texture_image[x,y] = c


    from PIL import Image
    plt.imsave("texture.png", texture_image)
    im = Image.open("texture.png")
    visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv,
        material=trimesh.visual.material.SimpleMaterial(
            image=im
        ),
        image=im
    )

    mesh.visual = visual
    mesh.export('tmp.obj')

    nvisii_mesh = visii.mesh.create_from_file(nv_gt+"mesh", 'tmp.obj')
    nvisii_tex = visii.texture.create_from_file(nv_gt+"tex",'material_0.png')
    visii_gt.set_mesh(nvisii_mesh)
    visii_gt.get_material().set_base_color_texture(nvisii_tex)
    visii_gu.get_transform().set_position(visii.vec3(-1000,-1000,-1000))

visii.render_to_file(
    width=im_width, 
    height=im_height, 
    samples_per_pixel=100,
    file_path=f'{opt.out}'
    # file_path=f'scene_colored/render_{scene_id}_{opt.bop_frame}_{opt.path_json_gu.split("/")[-2]}_{os.path.basename(opt.path_json_gu)}.png',
)

# # # ## #  # # # ## # 

visii.deinitialize()
