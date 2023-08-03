import torch
import numpy as np
import open3d as o3d
import os

#correspondence visualization
def draw_correspondence(p_pred, Obj, offset=[0,0,0], raw_CAD_down_sample=10000, models_path=None):
    """Visualize predicted correspondences between partial point cloud and CAD model.
    (requires desktop enviroment by o3d draw geometries)

    Args:
        p_pred (array): 2xN, predicted correspondences
        Obj: dict contains: align_pc, obj_id and diam_cad.
        offset: manually set vis offset, by default offset by mean coordinate distance
        raw_CAD_down_sample: The orignal mesh downsample parameter used in dataloader
        models_path: maunally set models path, by default find it automatically

    Returns:
        None
    """
    #find the original models path
    if not models_path:
        data_root = os.environ['data_root']
        render_data_name = os.environ['render_data_name']
        models_path = os.path.join(data_root, render_data_name, "models")

    #prep input varialbes
    p_pred = to_numpy_array(p_pred)
    align_pc = to_numpy_array(Obj["align_pc"])
    diam_cad = to_numpy_array(Obj['diam_cad'])
    obj_id = to_numpy_array(Obj['obj_id'])

    #load original cad to have CAD texture, but has to downsample again
    CAD_mesh = o3d.io.read_triangle_mesh(str(models_path+"/obj_0000"+"{:02d}".format(obj_id)+".ply"))
    CAD_mesh = CAD_mesh.simplify_quadric_decimation(raw_CAD_down_sample)
    CAD_verts = np.asarray(CAD_mesh.vertices)*0.1#this scale is from dataloader
    CAD_mesh.vertices = o3d.utility.Vector3dVector(CAD_verts)

    #offset prep
    if offset == [0,0,0]:
        offset = align_pc.mean(0) - CAD_verts.mean(0)
        offset = offset * -3

    #separate pred correspondences into inlier and outliers
    inlier, outlier = sep_in_out_lier(p_pred.T, CAD_verts, align_pc, 0.1*diam_cad)

    #change names for the code below
    inlier_lines = inlier
    outlier_lines = outlier

    #create o3d source-cad, target-pc instances
    source_points = o3d.geometry.PointCloud()
    source_points.points = CAD_mesh.vertices
    target_points = o3d.geometry.PointCloud()
    target_points.points = o3d.utility.Vector3dVector(align_pc - np.array(offset))

    # create colors for inliers and outliers
    inlier_colors = [[0, 1, 0] for _ in range(len(inlier_lines))]  # red color for inlier lines
    outlier_colors = [[1, 0, 0] for _ in range(len(outlier_lines))]  # green color for outlier lines

    # create line sets
    inlier_line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_points, target_points, inlier_lines)
    outlier_line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_points, target_points, outlier_lines)

    # assign colors
    inlier_line_set.colors = o3d.utility.Vector3dVector(inlier_colors)
    outlier_line_set.colors = o3d.utility.Vector3dVector(outlier_colors)

    #render point cloud as spheres only to look nicer
    spheres = []
    for point in np.asarray(target_points.points):
        # create a sphere of radius 0.01 at the point location
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(point)  # translate the sphere to the point location
        sphere.paint_uniform_color([0x64 / 255.0, 0x95 / 255.0, 0xED / 255.0])
        spheres.append(sphere)

    # Combine all the spheres into one mesh for more efficient visualization
    combined = spheres[0]
    for s in spheres[1:]:
        combined += s

    o3d.visualization.draw_geometries([CAD_mesh, combined, inlier_line_set, outlier_line_set])


#helper function
def sep_in_out_lier(pred_corr, CAD, PC_aligned, threshold):
    ''' given predicted correspondences, seperate out inlier and outlier pairs
    '''
    CAD = CAD[pred_corr[:,0]]
    PC_aligned = PC_aligned[pred_corr[:,1]]

    sq_dist = np.square(CAD - PC_aligned).sum(-1) ** 0.5

    inliers = np.argwhere((sq_dist<(threshold)))[:,0]

    inliers = pred_corr[inliers]
    outliers = []
    for p in pred_corr:
        if p in inliers:
            continue
        else:
            outliers.append(p)

    return inliers, np.array(outliers)

#helper function just to ensure converting whatever messy variable to np array
def to_numpy_array(x):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()  # Move to CPU if it's on CUDA
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("Input should be a PyTorch tensor or a NumPy array.")
