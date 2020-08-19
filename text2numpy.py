# To add a new cell, type '# %%'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
import argparse

def mkdirs(dir):
    dir=os.path.abspath(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

parser = argparse.ArgumentParser()
parser.add_argument("--branch", action="store", default="tree41")
parser.add_argument("--input_dir", action="store", default="gh_exported_64")
parser.add_argument("--size", action="store", default="64")

FLAGS = parser.parse_args()

input_dir = os.path.join(FLAGS.input_dir, FLAGS.branch)
output_dir = os.path.join("data", FLAGS.branch)

print("input", input_dir, "\n output", output_dir)



def main():
    ### check if it exits
    if os.path.exists(input_dir) is False:
        print("input dir ", input_dir, " does not exists")
        sys.exit()

    if os.path.exists(output_dir) is False:
        print("output dir ", output_dir, " does not exists")
        mkdirs(output_dir)

    if os.path.join(output_dir, "images") is False:
        os.mkdir(os.path.join(output_dir, "images"))
            

    fs = os.listdir(input_dir)
    fs.sort()
    filenames = []

    ### add filename in fs to filenames
    for filename in fs:
        # if(filename[-4:] == ".png"):
        #     print(filename)
        if("pattern_1" in filename):
            print(filename, '_'.join(filename.split("_")[:3]))
            filenames.append('_'.join(filename.split("_")[:3]))
        else:
            continue

    ## check if there is already files
    outputfiles = os.listdir(output_dir)
    print(outputfiles)
    for filename in filenames:
        size = int(FLAGS.size)
        print(filename)

        ### skip existing files
        if(filename + "_vol.npy" in outputfiles): 
            print(filename, " is already done!")
            continue

        try:
            path = os.path.join(input_dir, filename)
            path_1 = path + "_pattern_1.txt"
            path_2 = path +  "_pattern_2.txt"
            all_txt = open(path_1, 'r')
            mask_txt = open(path_2, 'r')
            # size = mask_txt.readline(0)
            vox_all = all_txt.readlines()[1].split(",")
            vox_mask = mask_txt.readlines()[1].split(",")
        except:
            print("failed to open" , filename,  path_1, path_2)
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
        print("pattern 1 is done", all_np.shape)

        ## culled pattern_2
        pattern_2 = []
        for vox in vox_mask:
            if(vox== '0'):
                pattern_2.append(0)
            elif(vox=='1'):
                pattern_2.append(1)
            else:
                print(vox)
        print(len(pattern_2))

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

        ax1 = fig.add_subplot(131, title='vol', projection='3d') 
        ax2 = fig.add_subplot(132, title='missing', projection='3d')
        ax3 = fig.add_subplot(133, title='masked volume', projection='3d') 
        # ax1 = fig.gca(projection='3d')
        ax1.voxels(all_np, facecolors='red', edgecolor='k')
        ax2.voxels(missing_parts, facecolors='green', edgecolor='k')
        ax3.voxels(masked_vol, facecolors='blue', edgecolor='k')

        fig_path = os.path.join(output_dir , filename + ".png")
        plt.savefig(fig_path) # get only the filename
        # plt.show()
        
        np.save(os.path.join(output_dir, filename + "_vol.npy"), all_np)
        np.save(os.path.join(output_dir, filename + "_missing.npy"), missing_parts)
        np.save(os.path.join(output_dir, filename + "_masked.npy"), masked_vol)
        print("saved ", os.path.join(output_dir, filename + "_vol.npy"))

if __name__ == '__main__':
    main()