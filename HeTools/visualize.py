import numpy as np
from os import path as osp
def show_result(
    points,
    gt_bboxes=None,
    pred_bboxes=None,
    out_dir=None,
    filename=None,
    show=False,
    snapshot=False,
    pred_labels=None,
    center_mode="lidar_center",
):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
        pred_labels (np.ndarray, optional): Predicted labels of boxes.
            Defaults to None.
    """
    if out_dir is not None:
        result_path = osp.join(out_dir, filename)
        hacv.mkdir_or_exist(result_path)

    if show:
        from .open3d_vis import Visualizer

        vis = Visualizer(points, center_mode=center_mode)
        if pred_bboxes is not None:
            if pred_labels is None or len(pred_bboxes) == 0:
                vis.add_bboxes(bbox3d=pred_bboxes)
            else:

                palette = (
                    np.array([[255, 0, 0], [0, 255, 0], [0, 255, 255], [192, 67, 251]])
                    / 256
                )
                if pred_labels.max() > 3:
                    np.random.seed(0)
                    palette_other = (
                        np.random.randint(0, 255, size=(pred_labels.max() + 1, 3)) / 256
                    )
                    palette = np.concatenate([palette, palette_other])
                palette = palette.clip(0, 1)

                labelDict = {}
                for j in range(len(pred_labels)):
                    if isinstance(pred_labels, np.ndarray):
                        i = int(pred_labels[j])
                    else:
                        i = int(pred_labels[j].numpy())
                    if labelDict.get(i) is None:
                        labelDict[i] = []
                    labelDict[i].append(pred_bboxes[j])
                for i in labelDict:
                    vis.add_bboxes(
                        bbox3d=np.array(labelDict[i]),
                        bbox_color=palette[i],
                        points_in_box_color=palette[i],
                    )

        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
        show_path = (
            osp.join(result_path, f"{filename}_online.png") if snapshot else None
        )
        vis.show(show_path)
