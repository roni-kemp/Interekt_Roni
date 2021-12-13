# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:57:27 2021

@author: YasmineMnb
"""
import h5py
import numpy as np
import hdf5utils
from matplotlib import pyplot as plt
import cv2
import pandas as pd

file_name = r"C:\Users\YasmineMnb\Desktop\transfer_folder\grv_testing\011221\interekt_data.h5"
#%%
dct = {}

with h5py.File(file_name, "r") as f:
#f = h5py.File(file_name, "r")

    for key in list(f.keys()):
        print(f"key:{key}")
        data = f[key]
        print(data)
        try:print(f"data:{list(data)}")
        except TypeError: print("cant... ")

#f.close()

#%%


my_file = hdf5utils.h5todict(file_name, path="data/tige0", exclude_names=None)
my_imgs = hdf5utils.h5todict(file_name, path="images", exclude_names=None)

#%%
def get_images_datetimes(hdf5file, img_pos=None):
    import datetime

    output_datetime = []
    output_fnames = []
    with h5py.File(hdf5file, 'r') as f:
        if img_pos is None:
            max_img = len(f['/images'].keys())
            for i in range(max_img):
                key = 'T%i'%i
                str_datetime = f['/images/'+key].attrs['datetime']
                f_name = f['/images/'+key].attrs['source_file']
                output_fnames.append(f_name)
                try:  # in case we get bytes
                    str_datetime = str_datetime.decode('ascii')
                except:
                    pass
                try:
                    output_datetime += [datetime.datetime.strptime(str_datetime,
                                                                   '%Y-%m-%d %H:%M:%S')]
                except:
                    print('No datetime for image %s, set it to None' % key)
                    output_datetime += [None]
        else:
            key = 'T%i' % img_pos
            str_datetime = f['/images/'+key].attrs['datetime'].decode('ascii')
            try:
                output_datetime = datetime.datetime.strptime(str_datetime,
                                                             '%Y-%m-%d %H:%M:%S')
            except:
                print('No datetime for image %s, set it to None' % key)
                output_datetime = None

    return output_datetime , max_img, output_fnames
#%%

with h5py.File(file_name, 'r') as f:
    if img_pos is None:
        max_img = len(f['/images'].keys())
        for i in range(max_img):
            key = 'T%i'%i
            print(list(f['/images/T0'].attrs))
#            print([f['/images/'+key].attrs['source_file']])

#%%

## ['base_points', 'diam', 'name', 'postprocessing',
## 'theta', 'xb1', 'xb2', 'xc', 'yb1', 'yb2', 'yc']
output_datetime, len_img_lst, output_fnames = get_images_datetimes(file_name, img_pos=None)

for img_i in range(len_img_lst)[::2]:
    print(img_i)
    #% looking at one img now
    fig = plt.figure()
    diam_array = my_file["diam"][img_i]
    thresh_diam = 160
    diam_lost = np.where(diam_array>thresh_diam)[0][0]

#    np.gradient(diam_array)
#    diam_lost += 200
    theta_array = my_file["theta"][img_i]

#    diam_lost = 1050
    yc =  my_file["yc"][img_i][:diam_lost]
    xc =  my_file["xc"][img_i][:diam_lost]
    plt.plot(xc, yc)

    base = (xc[0], yc[0])

    ## choos where to cutoff based on the change in diameter
    d_diam_thresh = 3
    diam_lost_der = np.where(abs(np.gradient(diam_array))>d_diam_thresh)[0][0]

    tip = (xc[diam_lost_der], yc[diam_lost_der])

    plt.plot(base[0], base[1], "o")
    plt.plot(tip[0], tip[1], "o")
    ## for finding different widths options
#    for i in range(5,10):
#        diam_lost_der = np.where(np.gradient(diam_array)>i)[0][0]
#        plt.plot(xc[diam_lost_der], yc[diam_lost_der], "o")

    img = cv2.imread(output_fnames[img_i].decode())
    plt.imshow(img)

#%%
    xb1 = my_file["xb1"][img_i]
    yb1 = my_file["yb1"][img_i]
    plt.plot(xb1, yb1, "o")
    plt.imshow(img)

#%%
#    q = np.where(abs(np.gradient(diam_array))>0.30)[0][0]
#    plt.plot(q, diam_array[1:diam_lost+50][q], "o")
#    plt.plot(diam_lost, diam_array[1:diam_lost+50][diam_lost], "o")
#    plt.plot(diam_array[1:diam_lost+50])
#    plt.plot(abs(np.gradient(diam_array[1:diam_lost+50])))
##    plt.plot(np.rad2deg(theta[img_i][:diam_lost]))
##    plt.plot(theta[img_i][1:diam_lost])
##    plt.plot(diam_lost, np.rad2deg(theta[img_i][diam_lost]), "o")
#    #%%
#    for i in range(0,500, 5):
#        theta_base= my_file["theta"][img_i][i]
#        print(i)
#        draw_line(tip, theta_tip, base, theta_base, img_i)


#%%
    #%%

    fig = plt.figure()
    plt.plot(diam_array)#[:diam_lost])
    plt.plot(abs(np.gradient(diam_array)[:diam_lost_der]))

#%% helper - draw line along angle from certain point

def draw_line(top_point, top_angle, base_point, base_angle, img_i):
#%
    ## Load img
    img = cv2.imread(output_fnames[img_i].decode())

    ## Calc the line representing the angle at the top
    size = 50
    top_angle_point = (int(top_point[0]-size*np.sin(top_angle)),
                       int(top_point[1]-size*np.cos(top_angle)))

    ## Calc the line for the base
    base_angle_point = (int(base_point[0]-size*np.sin(base_angle)),
                        int(base_point[1]-size*np.cos(base_angle)))

    ## Convert to integers
    base_point = tuple(int(i) for i in base_point)
    top_point = tuple(int(i) for i in top_point)

    ## Draw lines and points
    color = (255,0,0)
    thickness = 2
    cv2.line(img, base_point, base_angle_point, color, thickness)
    cv2.line(img, top_point, top_angle_point, color, thickness)

    cv2.circle(img, base_point, 5, color, -1)
    cv2.circle(img, top_point, 5, color, -1)

    cv2.imshow("showing img", img)
    cv2.waitKey(10)



#    ## Using plt
#    data_x = [start_point[0],end_point[0]]
#    data_y = [start_point[1],end_point[1]]
#    fig = plt.figure()
#    plt.plot(data_x, data_y)
#    plt.imshow(img)


    #%%


def collect_for_data_for_mu(file_name, output_data):
    """
    gets h5 file, saves data to output data csv file
    """

    ## load data from h5
    output_datetime, len_img_lst, output_fnames = get_images_datetimes(file_name, img_pos=None)

    ## init df for the relevent data
    df = pd.DataFrame(columns = ["dt_Sec", "theta_base_rad", "theta_tip_rad", "base_coor", "tip_coor"])


    for img_i in range(len_img_lst)[:15:2]:

        ## We need the diametere for cut off of the leaf
        diam_array = my_file["diam"][img_i]

        ## Choos where to cutoff based on the change in diameter
        d_diam_thresh = 3
        diam_lost = np.where(abs(np.gradient(diam_array))>d_diam_thresh)[0][0]
        mean_diam = int(np.mean(diam_array[:diam_lost]))

        ## Get x,y coordinates of the center
        yc =  my_file["yc"][img_i][:diam_lost]
        xc =  my_file["xc"][img_i][:diam_lost]

        ## Get x,y coordinates of tip and base.
        tip = (round(xc[-1],2), round(yc[-1],2))
        base = (round(xc[0],2), round(yc[0],2))

        ## Calc avg of last 2r data points as tip angle

        theta_tip = np.mean(my_file["theta"][img_i][diam_lost - mean_diam :diam_lost])
#        np.rad2deg(my_file["theta"][img_i][diam_lost])
        theta_base = np.mean(my_file["theta"][img_i][:mean_diam])


        ## calc dt
        dt = output_datetime[img_i] - output_datetime[0]
        dt = dt.seconds

        df.loc[img_i] = (dt, theta_base, theta_tip, base, tip)

    df.to_csv(output_data)
#%%


img = cv2.imread(output_fnames[img_i].decode())

draw_line(base, theta_base, img)

plt.plot(base[0], base[1], "o")
plt.plot(tip[0], tip[1], "o")

plt.plot(xc, yc, "o")
#%%
plt.plot(my_file["theta"][img_i][:diam_lost])


#%%


output_datetime
q = output_datetime[0]-output_datetime[1]
q.

