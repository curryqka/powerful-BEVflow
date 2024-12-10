import numpy as np

voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
height_feat_size = 13  # The size along the height dimension
cell_category_num = 5  # The number of object categories (including the background)
h,w,c = (area_extents[:, 1] - area_extents[:, 0]).astype(int)/voxel_size
out_seq_len = {
    'nuscenes': 20,
    'at128_0': 10,
    'wt128_0': 10,
}  # The number of future frames we are going to predict
trans_matrix_idx = 1  # Among N transformation matrices (N=2 in our experiment), which matrix is used for alignment (see paper)
params = dict(
    voxel_size=voxel_size,
    area_extents=area_extents,
    height_feat=height_feat_size,
    cell_category_num=cell_category_num,
    out_seq_len=out_seq_len,
    grid_h=int(h),
    grid_w=int(w),
)