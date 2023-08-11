import torch
import numpy as np
import open3d as o3d
import os
from PIL import Image
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import polyscope as ps


#usuage example 
#(Note: requires desktop env by the visualizer)

#mesh
#draw_basis(verts, faces, evecs, output="mesh_basis.png", evecs_selection=range(25,30), crop=[0.2, 0.1, 0.2, 0])

#point cloud
#draw_basis(verts, None, evecs, output="cloud_basis.png", evecs_selection=range(25,30), crop=[0.2, 0.1, 0.2, 0])

#draw_features(CAD, PC, Obj, C_pred[0], overlap_score12, overlap_score21, use_feat1[0], use_feat2[0], offset=[0,-18,0]) #assume input has no batch dim

#draw_correspondence(p_pred, Obj, offset=None, raw_CAD_down_sample=10000, models_path=None)

def draw_basis(verts, faces, evecs, output="eig_mesh.png", evecs_selection=range(3,8), crop=[0.2, 0.1, 0.2, 0]):
    """Visualize eigen basis functions on a mesh or a point cloud.
    
    first window pop up, adjust the camera view to desired, then close the window
    img of selected basis will be generated

    (requires desktop enviroment by polyscope)

    Args:
        verts: verts from the mesh, np array
        faces: faces from the mesh, np array, if leave empty or None, assume is point cloud
        evecs: evecs from the mesh, np array
        output(opt): output image file path
        evecs_selection(opt): pick the indexes of the selected eigen basis functions to visualize
        crop(opt): adjust the image crop ratio

    Returns:
        None
    """
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    if faces is None: #point cloud
        ps_mesh = ps.register_point_cloud("my cloud", to_numpy_array(verts))
    else:
        ps_mesh = ps.register_surface_mesh("my mesh", to_numpy_array(verts), to_numpy_array(faces), smooth_shade=True)
    evecs = to_numpy_array(evecs)
    
    #pop up window for you to adjust to desired camera view and point render radius etc before generating image
    ps.show()

    for i in evecs_selection:
        ps_mesh.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True, cmap='coolwarm')
        ps.screenshot(transparent_bg=False)

    # Collect images
    images = [Image.open(img) for img in sorted(glob.glob("screenshot_*.png"))]

    images = crop_images(images, *crop)

    # Concatenate images
    concat_image = Image.new('RGB', (images[0].width * len(images), images[0].height))

    # Loop over images to paste them together
    for i, img in enumerate(images):
        concat_image.paste(img, (i*images[0].width, 0))

    # Save the final image
    concat_image.save(output)
    
    # Delete original images
    images = [Image.open(img) for img in sorted(glob.glob("screenshot_*.png"))]
    for img in images:
        os.remove(img.filename)

#helper function
def crop_images(images, left_fraction, top_fraction, right_fraction, bottom_fraction):
    cropped_images = []
    for img in images:
        width, height = img.size
        left = width * left_fraction
        top = height * top_fraction
        right = width * (1 - right_fraction)
        bottom = height * (1 - bottom_fraction)
        cropped_images.append(img.crop((left, top, right, bottom)))
    return cropped_images



def draw_features(CAD, PC, Obj, C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2, offset=[0,-18,0]):
    """
    Visualize features and overlap scores on given CAD and point cloud. 
    (requires desktop env by polyscope)

    Note: strip away any batch dimension before use the function

    Parameters:
    - CAD (dict): 
        - 'xyz'
        - 'faces' 
    - PC (UNUSED)
    - Obj (dict): 
        - 'align_pc' : Aligned point cloud.
        - 'overlap_12' : Ground truth overlap scores for CAD.
        - 'overlap_21' : Ground truth overlap scores for point cloud.
    - C_pred (UNUSED)
    - overlap_score12 : Predicted overlap scores for CAD.
    - overlap_score21 : Predicted overlap scores for point cloud.
    - use_feat1 : Features associated with CAD for visualization.
    - use_feat2 : Features associated with point cloud for visualization.
    - offset : offset between CAD and point cloud

    """
    #initial setup
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    #define point cloud
    cloud = to_numpy_array(Obj['align_pc'])-offset
    ps_cloud = ps.register_point_cloud("my cloud", cloud)

    #define mesh
    verts = to_numpy_array(CAD['xyz'])
    faces = to_numpy_array(CAD['faces'])
    ps_mesh = ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)

    #add features colors
    comparable_pca = ComparablePCA(n_components=3)
    ps_cloud.add_color_quantity("features", comparable_pca.fit_transform(to_numpy_array(use_feat2)), enabled=True)
    ps_mesh.add_color_quantity("features", comparable_pca.transform(to_numpy_array(use_feat1)), enabled=True)

    #add overlap colors
    ps_mesh.add_scalar_quantity("overlap", to_numpy_array(overlap_score12), enabled=True, cmap='viridis')
    ps_cloud.add_scalar_quantity("overlap", to_numpy_array(overlap_score21), enabled=True, cmap='viridis')
    
    #add ground truth overlap
    ps_mesh.add_scalar_quantity("gt overlap", to_numpy_array(Obj['overlap_12']), enabled=True, cmap='viridis')
    ps_cloud.add_scalar_quantity("gt overlap", to_numpy_array(Obj['overlap_21']), enabled=True, cmap='viridis')
    
    
    ps.show()

class ComparablePCA:
    def __init__(self, n_components=3):
        self.pca = PCA(n_components=n_components)
        self.scaler = MinMaxScaler()
        
    def fit(self, X):
        """
        Fit the PCA on the input data and also fit the MinMaxScaler on the transformed data.
        """
        transformed_X = self.pca.fit_transform(X)
        self.scaler.fit(transformed_X)

    def transform(self, X):
        """
        Transform the input data using the previously fitted PCA and scaler.
        """
        transformed_X = self.pca.transform(X)
        scaled_X = self.scaler.transform(transformed_X)
        return scaled_X

    def fit_transform(self, X):
        """
        Fit the PCA and scaler on the input data, and then transform it.
        """
        self.fit(X)
        return self.transform(X)

#correspondence visualization
def draw_correspondence(p_pred, Obj, offset=None, raw_CAD_down_sample=10000, models_path=None):
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
    if offset is None:
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
    """
    Convert a PyTorch tensor or a NumPy array to a NumPy array. If the tensor requires gradient, 
    it will be detached before conversion.

    Parameters:
    - x (torch.Tensor or np.ndarray): Input tensor or array.

    Returns:
    - numpy_array (np.ndarray): Converted NumPy array.
    """
    if isinstance(x, torch.Tensor):
        # Detach from computation history if tensor requires gradient
        if x.requires_grad:
            x = x.detach()

        # Move to CPU if it's on CUDA
        if x.is_cuda:
            x = x.cpu()

        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("Input should be a PyTorch tensor or a NumPy array.")

