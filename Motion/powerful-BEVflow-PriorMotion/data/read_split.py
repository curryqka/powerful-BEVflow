import numpy as np
data_path = '/home/qiankangan/UniGenMotionNet/data/split.npy'
data = np.load(data_path, allow_pickle=True)
val_data_dict = data.item().get('val')
with open('/home/qiankangan/UniGenMotionNet/data/visual_val_list.txt', 'w') as f:
    for item in val_data_dict:
        item_id = item.split('_')[1]
        f.write(f"{item_id}\n")

