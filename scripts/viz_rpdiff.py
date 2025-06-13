import numpy as np
import os
# from non_rigid.utils.vis_utils import plot_diffusion
from PIL import Image
import pybullet as p
import matplotlib.pyplot as plt
from matplotlib import rcParams



# CAM_PROJECTION = (1.0, 0.0, 0.0, 0.0,
#                     0.0, 1.0, 0.0, 0.0,
#                     0.0, 0.0, -1.0000200271606445, -1.0,
#                     0.0, 0.0, -0.02000020071864128, 0.0)
# CAM_VIEW = (0.9396926164627075, 0.14454397559165955, -0.3099755346775055, 0.0, 
#             -0.342020183801651, 0.3971312642097473, -0.8516507148742676, 0.0, 
#             7.450580596923828e-09, 0.9063077569007874, 0.4226182699203491, 0.0, 
#             0.5810889005661011, -4.983892917633057, -22.852874755859375, 1.0)

CAM_PROJECTION = (1.299038052558899, 0.0, 0.0, 0.0, 
                    0.0, 1.7320507764816284, 0.0, 0.0, 
                    0.0, 0.0, -1.0020020008087158, -1.0, 
                    0.0, 0.0, -0.0200200192630291, 0.0)


CAM_VIEW = (0.7071068286895752, -0.29883620142936707, 0.6408563852310181, 0.0, 
            0.7071068286895752, 0.29883620142936707, -0.6408563852310181, 0.0, 
            1.4901161193847656e-08, 0.9063078761100769, 0.42261818051338196, 0.0, 
            -0.3535534739494324, -1.119412899017334, -1.5120935440063477, 1.0)

CAM_WIDTH = 640
CAM_HEIGHT = 480

def camera_project(point,
                   projectionMatrix,
                   viewMatrix,
                   height=CAM_HEIGHT,
                   width=CAM_WIDTH):
    """
    Projects a world point in homogeneous coordinates to pixel coordinates
    Args
        point: np.array of shape (N, 3); indicates the desired point to project
    Output
        (x, y): np.array of shape (N, 2); pixel coordinates of the projected point
    """
    point = np.concatenate([point, np.ones_like(point[:, :1])], axis=-1) # N x 4

    # reshape to get homogeneus transform
    persp_m = np.array(projectionMatrix).reshape((4, 4)).T
    view_m = np.array(viewMatrix).reshape((4, 4)).T

    # Perspective proj matrix
    world_pix_tran = persp_m @ view_m @ point.T
    world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
    world_pix_tran[:3] = (world_pix_tran[:3] + 1) / 2
    x, y = world_pix_tran[0] * width, (1 - world_pix_tran[1]) * height
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)

    return np.stack([x, y], axis=1)

def plot_diffusion(img, results, projmat, viewmat, color_key):
    # https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
    diffusion_frames = []

    for res in results:
        res_cam = camera_project(res, projectionMatrix=projmat, viewMatrix=viewmat)
        print(np.mean(res_cam, axis=0))

        dpi = rcParams['figure.dpi']
        img = np.array(img)
        height, width, _ = img.shape
        figsize = width / dpi, height / dpi

        # creating figure of exact image size
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img)

        # clipping points that are outside of camera frame
        filter = (res_cam[:, 0] >= 0) & (res_cam[:, 0] < width) & (res_cam[:, 1] >= 0) & (res_cam[:, 1] < height)
        res_cam = res_cam[filter]
        color_key_cam = color_key[filter]

        # plotting points
        ax.scatter(res_cam[:, 0], res_cam[:, 1], cmap="inferno", 
                   c=color_key_cam, s=5, marker='o')
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        diffusion_frames.append(frame)
    return diffusion_frames


eval_view = 0

path = "/home/lyuxing/Desktop/third_party/rpdiff/src/rpdiff/eval_data/eval_data/vis/mug_multi_med_rack/tax3dv2_fixed_9isflxpv_gmm1000_pfinetune_server/datamodule/seed_0"
pred_data = dict(np.load(os.path.join(path, "data", "demo_aug_27_pred.npz"), allow_pickle=True))

img = Image.open(os.path.join(path, "vis_imgs", f"27_{eval_view}_env.png")).convert("RGB")

results_world = pred_data["pred"].item()["results_world"]
results = [res.cpu().numpy().squeeze() / 15.0 for res in results_world]
pc_action = pred_data["batch"].item()["pc_action"].squeeze().cpu().numpy()

# create color key
color_key = -np.mean(pc_action, axis=1)

# get view matrix
yaw_angles = [45.0, 135.0, 225.0, 315.0]
distance = 0.6
pitch = -25.0
focus_pt = [0.5, 0.0, 1.4]

# view0 = p.computeViewMatrixFromYawPitchRoll(
#     cameraTargetPosition = focus_pt,
#     distance = distance,
#     yaw = yaw_angles[0],
#     pitch = pitch,
#     roll = 0,
#     upAxisIndex = 2,
# )
# view1 = p.computeViewMatrixFromYawPitchRoll(
#     cameraTargetPosition = focus_pt,
#     distance = distance,
#     yaw = yaw_angles[1],
#     pitch = pitch,
#     roll = 0,
#     upAxisIndex = 2,
# )
# view2 = p.computeViewMatrixFromYawPitchRoll(
#     cameraTargetPosition = focus_pt,
#     distance = distance,
#     yaw = yaw_angles[2],
#     pitch = pitch,
#     roll = 0,
#     upAxisIndex = 2,
# )
# view3 = p.computeViewMatrixFromYawPitchRoll(
#     cameraTargetPosition = focus_pt,
#     distance = distance,
#     yaw = yaw_angles[3],
#     pitch = pitch,
#     roll = 0,
#     upAxisIndex = 2,
# )

def get_view(yaw):
    return p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition = focus_pt,
        distance = distance,
        yaw = yaw,
        pitch = pitch,
        roll = 0,
        upAxisIndex = 2,
    )
view0 = get_view(yaw_angles[0])
view1 = get_view(yaw_angles[1])
view2 = get_view(yaw_angles[2])
view3 = get_view(yaw_angles[3])

# _C.ZNEAR = 0.01
# _C.ZFAR = 10
# _C.WIDTH = 640
# _C.HEIGHT = 480
# _C.FOV = 60

projection_matrix = p.computeProjectionMatrixFOV(
    fov = 60,
    aspect=  640.0/480.0,
    nearVal = 0.01,
    farVal = 10,
)

frames0 = plot_diffusion(img, results, projection_matrix, view0, color_key)
frames1 = plot_diffusion(img, results, projection_matrix, view1, color_key)
frames2 = plot_diffusion(img, results, projection_matrix, view2, color_key)
frames3 = plot_diffusion(img, results, projection_matrix, view3, color_key)



frames0[0].save(f"/home/lyuxing/Desktop/view{eval_view}frame0.gif", save_all=True,
    append_images=frames0[1:], duration=33, loop=0)
frames1[0].save(f"/home/lyuxing/Desktop/view{eval_view}frame1.gif", save_all=True,
    append_images=frames1[1:], duration=33, loop=0)
frames2[0].save(f"/home/lyuxing/Desktop/view{eval_view}frame2.gif", save_all=True,
    append_images=frames2[1:], duration=33, loop=0)
frames3[0].save(f"/home/lyuxing/Desktop/view{eval_view}frame3.gif", save_all=True,
    append_images=frames3[1:], duration=33, loop=0)

breakpoint()