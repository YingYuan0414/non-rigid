import numpy as np
import os
import rpad.visualize_3d.plots as vpl

train_dir = "/home/eycai/data/learn_data/train"
test_dir = "/home/eycai/data/learn_data/test"

# iterate through the files in the directory
for filename in os.listdir(train_dir):
    print(filename)
    data = np.load(os.path.join(train_dir, filename), allow_pickle=True)
    data = dict(data)

    classes = data["classes"]
    action = data["clouds"][classes == 0]
    anchor = data["clouds"][classes == 1]

    fig = vpl.segmentation_fig(
        data["clouds"],
        data["classes"].astype(int),
    )
    fig.show()
    breakpoint()
    # if filename.endswith(".ply"):
    #     # load the point cloud
    #     pcd = o3d.io.read_point_cloud(os.path.join(train_dir, filename))
    #     # visualize the point cloud
    #     o3d.visualization.draw_geometries([pcd])