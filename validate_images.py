import os
from PIL import Image 
import pickle
from multiprocessing import Pool


rootdir = '/vol/bitbucket/pv819/sceneflow_data'
corrupted_file_lst = []

def process_file(file):
    try:
        Image.open(file).convert('RGB')
    except OSError:
        print(f"OSError in file {file}")
        corrupted_file_lst.append(file)


# file_lst = []

# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         if file.endswith('.png'):
#             file_lst.append(os.path.join(subdir, file))
#             # print(f"Found img {os.path.join(subdir, file)}")
#             # try:
#             #     Image.open(os.path.join(subdir, file)).convert('RGB')
#             # except OSError:
#             #     print(f"OSError in file {os.path.join(subdir, file)}")

# with open('filelist', 'wb') as fp:
#     pickle.dump(file_lst, fp)

with open('filelist', 'rb') as f:
    file_lst = pickle.load(f)

print(len(file_lst))
print(file_lst[0])

p = Pool(12)
p.map(process_file, file_lst) 
p.close()
p.join()
with open('curruptedfilelist', 'wb') as fp:
    pickle.dump(corrupted_file_lst, fp)