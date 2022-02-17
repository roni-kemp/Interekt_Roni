# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:34:50 2021

@author: YasmineMnb
"""
import h5py
import numpy as np
import pandas as pd
import cv2
from scipy.signal import butter,filtfilt

# =============================================================================
# so that i don't need to be in the rootstem extractor dir i took some functions
# import hdf5utils
# =============================================================================
##

def _name_contains_string_in_list(name, strlist):
    if strlist is None:
        return False
    for filter_str in strlist:
        if filter_str in name:
            return True
    return False

class _SafeH5FileReadWrite(object):
    """
    Context manager returning a :class:`h5py.File` object.
    """
    def __init__(self, h5file, mode="w"):
        """
        """
        self.raw_h5file = h5file
        self.mode = mode

    def __enter__(self):
        if not isinstance(self.raw_h5file, h5py.File):
            self.h5file = h5py.File(self.raw_h5file, self.mode)
            self.close_when_finished = True
        else:
            self.h5file = self.raw_h5file
            self.close_when_finished = False
        return self.h5file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.close_when_finished:
            self.h5file.close()

def h5todict(h5file, path="/", exclude_names=None):
    """Read a HDF5 file and return a nested dictionary with the complete file

    From hdf5utils
    """
    ddict = None
    with _SafeH5FileReadWrite(h5file, mode='r') as h5f:
        if path in h5f and isinstance(h5f[path], h5py.Group):
            ddict = {}
            for key in h5f[path]:
                if _name_contains_string_in_list(key, exclude_names):
                    continue
                if isinstance(h5f[path + "/" + str(key)], h5py.Group):
                    ddict[key] = h5todict(h5f,
                                          path + "/" + str(key),
                                          exclude_names=exclude_names)
                else:
                    # Convert HDF5 dataset to numpy array
                    ddict[key] = h5f[path + "/" + str(key)][...]
        else:
            if path in h5f:
                ddict = h5f[path][...]
                if ddict.shape == ():
                    ddict = ddict.tolist()

    return ddict

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



def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# =============================================================================
#  my code...
# =============================================================================

def draw_line(top_point, top_angle, base_point, base_angle, img_i, output_fnames):
#%
    ## Load img
    img = cv2.imread(output_fnames[img_i].decode())
    
    try:
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
        cv2.waitKey(1)
    except ValueError:
        print("bad img")
#%
    return img

def save_vid(img_lst, out_path):
    ## set params for the vid_output

    outvid_path = out_path.replace(".csv", ".mp4")
    size = img_lst[0].shape
    # size = (int(size[0]/2),int(size[1]/2))
    print(size)
    is_color = True
    fps = 12.0
#    fourcc = cv2.VideoWriter_fourcc(*"XVID") ## .avi
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  ## .mp4
    vid = cv2.VideoWriter(outvid_path, fourcc, fps, (size[1], size[0]), is_color)
    size = size[:2]
    for img in img_lst:
        ## write to video file
        # img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
        vid.write(img)
        # cv2.imshow("showing img", img)
        # cv2.waitKey(1)
    vid.release()
#%%
def collect_for_data_for_mu(file_name, output_data):
    """
    gets h5 file, saves data to output data csv file
    """
#%

    ## load data from h5
    my_file = h5todict(file_name, path="data/tige0", exclude_names=None)

    output_datetime, len_img_lst, output_fnames = get_images_datetimes(file_name, img_pos=None)

    ## init df for the relevent data
    df = pd.DataFrame(columns = ["dt_Sec", "theta_base_rad", "theta_tip_rad", "base_coor", "tip_coor"])

    img_lst = []

    for img_i in range(50, len_img_lst):
#%
        ## We need the diametere for cut off of the leaf
        diam_array = my_file["diam"][img_i]
        if diam_array[0] == 30000:
            continue
        print(img_i)
        if len(diam_array)<=200:
            print("error\n\n\n")
    
        diam_array = butter_lowpass_filter(diam_array, 1, 20, 2)
        
        try: 
            diam_lost = np.where(abs(np.gradient(diam_array[200:]))>0.34)[0][0] + 200
        except IndexError:
            continue
        
        # if diam_lost < 500:
        #     diam_lost = np.where(abs(np.gradient(diam_array[400:]))>0.34)[0][0] + 400

        mean_diam = int(np.mean(diam_array[:diam_lost]))*3

        ## Get x,y coordinates of the center
        yc =  my_file["yc"][img_i][:diam_lost]
        xc =  my_file["xc"][img_i][:diam_lost]

        ## Get x,y coordinates of tip and base.
        tip = (round(xc[-1],2), round(yc[-1],2))
        base = (round(xc[0],2), round(yc[0],2))

        ## dict_keys(['angle', 'angle_au_bout', 'angle_zone_de_mesure', 'smooth_s', 'smooth_xc', 'smooth_yc', 'taille'])
        theta = my_file["postprocessing"]["angle"]
#        theta = my_file["theta"]

        ## Calc avg of last 2r data points as tip angle
        theta_tip = np.mean(theta[img_i][diam_lost - mean_diam :diam_lost])
        theta_base = np.mean(theta[img_i][1:mean_diam])

        
        img = draw_line(tip, theta_tip, base, theta_base, img_i, output_fnames)

        try:
            for i in range(diam_lost):
                point = (int(my_file["xc"][img_i][i]), int(my_file["yc"][img_i][i]))
                cv2.circle(img, point, 1, (200,100,200), -1)
    
            for i in range(mean_diam):
                point = (int(my_file["xc"][img_i][i]), int(my_file["yc"][img_i][i]))
                cv2.circle(img, point, 2, (55,50,255), -1)
    
            for i in range(diam_lost - mean_diam, diam_lost):
                point = (int(my_file["xc"][img_i][i]), int(my_file["yc"][img_i][i]))
                cv2.circle(img, point, 2, (0,200,0), -1)
        except IndexError:
            pass
        cv2.imshow("showing img", img)
        k = cv2.waitKey(10) & 0xff
        if k == 27 or k == ord('q'):
            break
        
        img_lst.append(img)

        ## calc dt
        dt = output_datetime[img_i] - output_datetime[0]
        dt = dt.seconds

        df.loc[img_i] = (dt, theta_base, theta_tip, base, tip)
#%
    ## Close img window
    cv2.destroyAllWindows()
    
    df.to_csv(output_data)
    print("saved to csv file...")

    save_vid(img_lst, output_data)
    print(f"saved video with {len(img_lst)} frames ({(len(img_lst)*4)/60}H)")

#%%
file_name = r"C:\Users\Roni\Desktop\transfer folder\bi 350 v 300\220206\IR\1__Croped_4\interekt_data.h5"
output_data =  r"C:\Users\Roni\Desktop\transfer folder\bi 350 v 300\220206\IR\1__Croped_4\out_test_2.csv"
#%
collect_for_data_for_mu(file_name, output_data)

# #%%
# for i in range(1,6):
#     if i == 20:
#         continue
#     else:
#         file_name = r"C:\Users\Roni\Desktop\grv_testing\011221\set1\interekt_data_{}.h5".format(i)
#         output_data =  r"C:\Users\Roni\Desktop\grv_testing\011221\set1\out_test_{}.csv".format(i)
#         collect_for_data_for_mu(file_name, output_data)
#%%% junk!


# from matplotlib import pyplot as plt
# plt.plot(abs(np.gradient(diam_array[200:])))
# plt.show()
