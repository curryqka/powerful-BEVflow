import pickle

# Replace 'your_file_path.pkl' with the path to your pickle file
file_path = '/mnt/yrfs/Team/Semantic/datasets/AllData/HaData/groundtruth/at128_0/Det_data_infos/2023_0209_1000_24.pkl'
file_path = '/mnt/yrfs/Team/Semantic/datasets/AllData/HaData/groundtruth/at128_0/Det_data_infos/task_10_30-13_38_13_4.db3_seq_9.pkl'
# Open the file in read-binary mode ('rb')
with open(file_path, 'rb') as file:
    # Load the pickle file
    data = pickle.load(file)

# Now, 'data' contains the contents of the pickle file

print(data)