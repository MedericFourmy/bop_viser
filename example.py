import argparse
from pathlib import Path
import random
import math

# 3rd party
import transforms3d
import distinctipy
import trimesh
import viser
from bop_toolkit_lib import inout
from bop_toolkit_lib import config
from bop_toolkit_lib import misc
from bop_toolkit_lib import dataset_params


"""
Demo script displaying some bop ground truth data using the viser library.

Notes:
- Use self corrective frame convention: T_13 = T_12 * T_23 
- Convert everything to meters (*0.001): meshes, poses, depth map
- Display single view estimates as meshes using 
"""

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--scene_id", type=int, default=48)
parser.add_argument("--ds_name", type=str, default="ycbv")
parser.add_argument("--ds_split", type=str, default="test")
parser.add_argument("-N", "--number_of_views", type=int, default=4)
parser.add_argument("--display_textured_mesh", action="store_true", default=False, help="textured mesh seem slower to load in viser")
args = parser.parse_args()

rng = random.Random(args.seed)
datasets_path = Path(config.datasets_path)
# select a few views to represent single view estimates
scene_id = args.scene_id

ds_split = "test"
targets_name = "test_targets_bop19.json"
targets = inout.load_json(datasets_path / args.ds_name / targets_name)
targets_org = misc.reorganize_targets(targets)
im_ids = rng.sample(list(targets_org[scene_id].keys()), args.number_of_views)
dp_split = dataset_params.get_split_params(datasets_path, args.ds_name, ds_split)
model_type = None  # basic textured one
model_type = "eval"  # smaller, no texture
dp_models = dataset_params.get_model_params(datasets_path, args.ds_name)

# get paths to view files (gt, rgb, intrinsics)
tpath_keys = dataset_params.scene_tpaths_keys(None, None, scene_id)
scene_camera = inout.load_scene_camera(dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id))
scene_gt = inout.load_scene_gt(dp_split[tpath_keys["scene_gt_tpath"]].format(scene_id=scene_id))


viser_server = viser.ViserServer()
viser_server.scene.add_grid("/ground")
colors = distinctipy.get_colors(args.number_of_views)
meshes = {}
for i, im_id in enumerate(im_ids):
    # cameras
    im_camera = scene_camera[im_id]
    cam_K = im_camera["cam_K"]
    R_cw, t_cw = im_camera["cam_R_w2c"], im_camera["cam_t_w2c"].flatten()*0.001
    R_wc, t_wc = R_cw.T, -R_cw.T@t_cw
    depth_scale = im_camera["depth_scale"]

    # images
    rgb_path = dp_split[tpath_keys["rgb_tpath"]].format(scene_id=scene_id, im_id=im_id)
    rgb = inout.load_im(rgb_path)
    depth_path = dp_split[tpath_keys["depth_tpath"]].format(scene_id=scene_id, im_id=im_id)
    depth = inout.load_depth(depth_path)*depth_scale*0.001

    # Viser: display camera as a nice frustrum
    h, w, _  = rgb.shape
    fy = cam_K[1,1]
    q_wc = transforms3d.quaternions.mat2quat(R_wc)  # quaternion expressed in wxyz convention
    params_frust = {
        "wxyz": q_wc, 
        "position": t_wc, 
        "scale": 0.05, 
        "fov": 2 * math.atan(h / (2 * fy)), 
        "aspect": w / h,
        "image": rgb,
        "color": colors[i],
    }
    viser_server.scene.add_camera_frustum(f"/camera_gt_{i}", **params_frust)

    # object poses in camera frame
    im_gts = scene_gt[im_id]
    for gt in im_gts:
        R_cm, t_cm =  gt["cam_R_m2c"] , gt["cam_t_m2c"].flatten()*0.001
        obj_id = gt["obj_id"] 
        if obj_id not in meshes:
            meshes[obj_id] = trimesh.load(dp_models["model_tpath"].format(obj_id=obj_id))
            meshes[obj_id].apply_scale(0.001)

        R_wm, t_wm = R_wc@R_cm, t_wc + R_wc @ t_cm
        q_wm = transforms3d.quaternions.mat2quat(R_wm)  # quaternion expressed in wxyz convention

        if args.display_textured_mesh:
            # display meshes with textures but cannot change opacity
            viser_server.scene.add_mesh_trimesh(
                f"/im_{im_id:06d}_obj_{obj_id:06d}",
                meshes[obj_id],
                wxyz=q_wm,
                position=t_wm,
            ) 
        else:
            # OPTION 2: no texture but can change opacity and color
            viser_server.scene.add_mesh_simple(
                name=f"/im_{im_id:06d}_obj_{obj_id:06d}",
                vertices=meshes[obj_id].vertices,
                faces=meshes[obj_id].faces,
                wxyz=q_wm,
                position=t_wm,
                opacity=0.7,
                color=colors[i]
            )