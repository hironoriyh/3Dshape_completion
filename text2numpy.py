# To add a new cell, type '# %%'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


input_dir = "./test/tree5/"
output_dir = "./data/tree5/"

fs = os.listdir(input_dir)
fs.sort()
filenames = []
for filename in fs:
    # if(filename[-4:] == ".png"):
    #     print(filename)
    if("pattern_1" in filename):
        print(filename, '_'.join(filename.split("_")[:4]))
        filenames.append('_'.join(filename.split("_")[:4]))
    else:
        continue

## check if there is already files
fs = os.listdir(output_dir)
# fs.sort()
print(fs)

for filename in filenames:
    size = 32
    print(filename)
    if(filename + "_all.npy" in fs): 
        print(filename, " is already done!")
        continue

    try:
        all_txt = open(input_dir + filename + "_pattern_1.txt", 'r')
        mask_txt = open(input_dir + filename + "_pattern_2.txt", 'r')
        # size = mask_txt.readline(0)
        vox_all = all_txt.readlines()[1].split(",")
        vox_mask = mask_txt.readlines()[1].split(",")
    except:
        print("failed to open" , filename)
        continue

    ## culled pattern_1
    all_np = []
    for vox in vox_all:
        # print(i, vox)
        if(vox== '0'):
            all_np.append(0)
        elif(vox=='1'):
            all_np.append(1)
        else:
            print(vox)
    all_np = np.array(all_np, dtype=np.uint8)
    all_np = np.reshape(all_np, (size,size,size))

    ## culled pattern_2
    pattern_2 = []
    for vox in vox_mask:
        if(vox== '0'):
            pattern_2.append(0)
        elif(vox=='1'):
            pattern_2.append(1)
        else:
            print(vox)
    # print(len(pattern_2))

    culled_idx = 0
    culled_list = []
    for n, i in enumerate(all_np.ravel()):
        if(i == 1): 
            culled_list.append(n)

    culled_list_index = 0
    masked_vol = all_np.ravel().copy()
    # print(masked_vol.shape)
    missing_parts = np.zeros(size**3)

    for i, vox in enumerate(masked_vol):
        if any(i == c for c in culled_list):
            if(pattern_2[culled_list_index] == 1):
                # print("missing parts" , i, "masked vol", masked_vol[i])
                missing_parts[i] = 1
                masked_vol[i] = 0
            culled_list_index +=1

    # print(missing_parts.shape, masked_vol.shape)
    missing_parts = np.reshape(missing_parts, (size,size,size))
    masked_vol = np.reshape(masked_vol, (size,size,size))
    print("shapes of missing parts and masked_vol", missing_parts.shape, masked_vol.shape)

    fig = plt.figure()
    fig = plt.figure(figsize=plt.figaspect(0.3))
    fig.suptitle(filename[0], fontsize=16)

    ax1 = fig.add_subplot(131, title='all', projection='3d') 
    ax2 = fig.add_subplot(132, title='missing', projection='3d')
    ax3 = fig.add_subplot(133, title='masked volume', projection='3d') 
    # ax1 = fig.gca(projection='3d')
    ax1.voxels(all_np, facecolors='red', edgecolor='k')
    ax2.voxels(missing_parts, facecolors='green', edgecolor='k')
    ax3.voxels(masked_vol, facecolors='blue', edgecolor='k')
    fig_path = output_dir + filename +".png"
    plt.savefig(fig_path) # get only the filename
    # plt.show()

    np.save(output_dir + filename + "_all.npy", all_np)
    np.save(output_dir + filename + "_missing.npy", missing_parts)
    np.save(output_dir + filename + "_masked.npy", masked_vol)
