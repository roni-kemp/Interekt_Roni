# -*- coding: utf-8 -*-
"""
Portagee de la detection de tige de Olivier (en matlab) vers python

Regroupe les differentes fonctions pour le traitements des images de la manip
du gravitron.

Version: 3.1
Date: 19/04/2018

Modif
-----
13/10/2016 Hugo: Clean the file libgravimacro to only keep one method and usefull functions

"""
import platform
if (platform.system().lower() == "windows"):
    _windows = True
else:
    _windows = False
    
try:
    # Pour charger les image rapidement
    import cv2
    _iscv2 = True
except:
    print('OpenCv is not installed, image loading will be slower')
    _iscv2 = False

# Imports des bibliothéques
from numpy import (array, asarray, uint8, ma, arctan2, sqrt,
                   gradient, sign, ctypeslib, ones, vstack, hstack,
                   arange, sin, cos, diff, linspace, tan, convolve,
                   cumsum, rad2deg, zeros, zeros_like, pi, log, exp,
                   deg2rad, warnings, arange)

# Remove Numpy Warning sur les masked array (attention potentiel bug avec les
# futurs version de Numpy > 1.11) changement de mode de copy du mask
warnings.filterwarnings('ignore')

from pylab import find, where
# from scipy.signal import argrelmax
from scipy.stats import circmean  # Pour faire des moyenne d'angle correcte!!!
from scipy import ndimage
from scipy import signal
import matplotlib.pylab as mpl

import glob
import cPickle as pickle
import time
import os

try:
    from IPython.display import HTML, Javascript, display, clear_output

    cfg = get_ipython().config
    if 'InteractiveShell' in cfg:
        _isnotebook = False
    else:
        _isnotebook = True
except:
    _isnotebook = False

from skimage.exposure import (adjust_sigmoid)
from skimage.morphology import disk
import multiprocessing as mp
import ctypes

# Pour la methode Hugo
# Pour charger les photos
try:
    from PIL import Image as pilimage
except:
    import Image as pilimage

import datetime
import re
from scipy.misc import fromimage
import rootstem_hdf5_store as h5store
from rootstem_hdf5_store import TigesManager, get_photo_time

import traceback
# Regexp pour trouver des nombres dans une ligne de texte
finddigits = re.compile(r'\d+?')


class Image():
    """
    Class pour gerer la façon dont on ouvre les image, elle permet
    de réunir opencv ou PIL avec les même méthodes pour ouvrir les
    images lors du traitement
    
    Paramètres:
    - fname: le nom de l'image a ouvrir ou son adresse dans le fichier hdf5
    - use_bw: doit on utilise du noir et blanc 
    - color_transform: doit on changer l'espace des couleurs, utiliser par opencv (expl: cv2.COLOR_BGR2RGB)
    - color_band: doit utiliser une bande (R->0,G->1,B->2) particulière lors de l'ouverture
    - maxwidth: on donne la taille max de l'image, si plus grande elle est réduite 
    - h5file: si on utilise le hdf5, c'est le nom du fichier qui contient les images 

    Expl avec param par defaut
    Image_reader().load( name )

        Option

            band

            color_converter:

            cv2.COLOR_BGR2LAB


    """

    def __init__(self, fname=None, use_bw=False, color_transform=None,
                 color_band=None, maxwidth=None, h5file=None):

        self.data = None
        self.use_bw = use_bw
        self.color_transform = color_transform
        self.color_band = color_band
        self.maxwidth = maxwidth
        self.ratio = 1.0
        self.orientation = 1
        self.h5file = h5file
        
        # Check if cv2 is loaded to open images
        if _iscv2:
            self.loader = cv2.imread
        else:
            self.loader = pilimage.open

        # Doit on charger les données d'une image dès la création de la classe    
        if fname is not None:
            self.load(fname)

    def load_img_part(self, fname, crop):
        """
        Petite fonction pour ne charger qu'une partie des images
        """
        
        if self.h5file is not None:
            imgn = h5store.image_to_bytesio(self.h5file, fname, resolution=0)
        else:
            imgn = fname

            
        imgobj = pilimage.open(imgn)    
        imgc = imgobj.crop(crop).convert('L')

        if self.use_bw:
            return fromimage(imgc) / 255.0
        else:
            return fromimage(imgc, 'gray') / 255.0

    def load(self, fname):
        """ 
        Charge l'image "fname" ou le chemin h5 "fname" en mémoire
        """
        
        if self.use_bw:
            if _iscv2:
                # Test si on doit utilise un fichier h5
                if self.h5file is None:
                    self.data = self.loader(fname, 0)

                else:
                    # Cas du fichier hdf5
                    imgio = h5store.image_to_bytesio(self.h5file, fname, resolution=0)
                    img_array = asarray(bytearray(imgio.read()), dtype=uint8)
                    self.data = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
   
            else:
                if self.h5file is None:
                    self.data = self.loader(fname).convert('L')
                else:
                    imgio = h5store.image_to_bytesio(self.h5file, fname, resolution=0)
                    self.data = self.loader(imgio).convert('L')

        else:
            if self.h5file is None:
                self.data = self.loader(fname)
                
            else:
                imgio = h5store.image_to_bytesio(self.h5file, fname, resolution=0)
                if _iscv2:
                    img_array = asarray(bytearray(imgio.read()), dtype=uint8)
                    self.data = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)
                else:
                    self.data = self.loader(imgio)
                    
        if self.maxwidth is None:
            if _iscv2:
                self.maxwidth = self.data.shape[0]
            else:
                self.maxwidth = self.data.size[0]

    def render(self, rescale=True):

        if _iscv2:
            # Check if we need to resize
            if self.maxwidth < self.data.shape[0]:
                self.resize()
            # Check if we need to transform color
            if self.color_transform is not None:
                out = cv2.cvtColor(self.data, getattr(cv2, self.color_transform))
            else:
                out = self.data

            # Check if we have select a specific image band
            if self.color_band is not None:
                out = out[:, :, self.color_band]
        else:
            if self.maxwidth < self.data.size[0]:
                self.resize()

            out = array(self.data)

        if rescale:
            return out / 255.
        else:
            return out

    def is_bw(self):
        out = False
        if _iscv2:
            if len(self.data.shape) > 2:
                out = True
        else:
            if self.data.mode == 'L':
                out = True

        return out

    def resize(self):
        if _iscv2:
            self.ratio = self.maxwidth / float(self.data.shape[0])
            self.data = cv2.resize(self.data, (int(self.ratio * self.data.shape[1]), int(self.maxwidth)))
        else:
            self.ratio = self.maxwidth / float(self.data.size[0])
            self.data = self.data.resize((int(self.maxwidth), int(self.ratio * self.data.size[1])))

    def __repr__(self):
        return self.render()


class Tiges():
    def __init__(self):
        print('Pour compatibilité avec le fichier root_stemdata.pkl des anciennes versions < 2017')
        
class TigesNew():

    def __init__(self, nbtige, size=2000):
        """
            Class pour stocker les tiges d'une image

            nbtige (int): number of tige inside the picture
            size: (int) [2000]: the number of step to store for the
        """

        self.size = size
        modeltab = ones([nbtige, size]) * 30000

        self.diam = ma.masked_equal(modeltab, 30000)
        self.xc = ma.masked_equal(modeltab, 30000)
        self.yc = ma.masked_equal(modeltab, 30000)
        self.theta = ma.masked_equal(modeltab, 30000)
        self.xb1 = ma.masked_equal(modeltab, 30000)
        self.yb1 = ma.masked_equal(modeltab, 30000)
        self.xb2 = ma.masked_equal(modeltab, 30000)
        self.yb2 = ma.masked_equal(modeltab, 30000)

        """
        for i in [self.diam, self.xc, self.yc, self.theta, self.xb1, self.yb1, self.xb2, self.yb2]:
            i.unshare_mask()
        """

    def add_point(self, nbimage, id_tige, pos_in_tige, diam, xc, yc, theta, xb1, yb1, xb2, yb2, graylevel):
        """
            Add values to the structure
        """

        self.diam[id_tige, pos_in_tige] = diam
        self.xc[id_tige, pos_in_tige] = xc
        self.yc[id_tige, pos_in_tige] = yc
        self.theta[id_tige, pos_in_tige] = theta
        self.xb1[id_tige, pos_in_tige] = xb1
        self.yb1[id_tige, pos_in_tige] = yb1
        self.xb2[id_tige, pos_in_tige] = xb2
        self.yb2[id_tige, pos_in_tige] = yb2

    def Mask_invalid(self):

        self.diam = ma.masked_equal(self.diam, 30000)
        self.xc = ma.masked_equal(self.xc, 30000)
        self.yc = ma.masked_equal(self.yc, 30000)
        self.theta = ma.masked_equal(self.theta, 30000)
        self.xb1 = ma.masked_equal(self.xb1, 30000)
        self.yb1 = ma.masked_equal(self.yb1, 30000)
        self.xb2 = ma.masked_equal(self.xb2, 30000)
        self.yb2 = ma.masked_equal(self.yb2, 30000)

    def compress_data(self):
        """
            Function to reduce array dimention to maximum point extracted in images
        """

        # Find the maximum of unmasked data using the first non-null sum on xc data
        try:
            iend_data = find(self.xc.sum(axis=(0, 1)).mask == True)[0]
        except:
            iend_data = 0

        if iend_data > 0:
            # Loop over data to reduce them
            self.diam = self.diam[:, :iend_data]
            self.xc = self.xc[:, :iend_data]
            self.yc = self.yc[:, :iend_data]
            self.theta = self.theta[:, :iend_data]
            self.xb1 = self.xb1[:, :iend_data]
            self.xb2 = self.xb2[:, :iend_data]
            self.yb1 = self.yb1[:, :iend_data]
            self.yb2 = self.yb2[:, :iend_data]


def get_tige_border(xi, yi, image, seuil_coupure=0.1):
    """
        Obtenir les bords de la tige a partir du profil

    """

    # zi = ndimage.map_coordinates(image.T, vstack( (xi, yi) ), order=1 )
    # cv2.INTER_LINEAR
    if _iscv2:
        zi = cv2.remap(image, xi, yi, cv2.INTER_LINEAR)[:, 0]
    else:
        zi = ndimage.map_coordinates(image.T, vstack((xi, yi)), order=1)

    # figure('line')
    # plot( zi )
    ib1, ib2 = get_min_max(zi, coupure=seuil_coupure)
    if type(ib1) == type(None) or type(ib2) == type(None):
        xcenterf = 30000.0
        ycenterf = 30000.0
        diam = 30000.0
        theta = 30000.0
        b1f = 30000.0
        b2f = 30000.0
        cgray = 30000
    else:

        # Calcul du centre et du rayon
        xb1, yb1 = xi[ib1], yi[ib1]
        xb2, yb2 = xi[ib2], yi[ib2]

        xcenterf, ycenterf = 0.5 * (xb1 + xb2), 0.5 * (yb1 + yb2)
        diam = sqrt((xb2 - xb1) ** 2 + (yb1 - yb2) ** 2)

        # Get the level of gray at center
        # ixc, iyc = int(xcenterf), int(ycenterf)
        # npix = int(diam/2) #taille du rectangle pour faire la moyenne sur les pixels
        # if npix > 2:
        #    cgray = image[iyc-npix/2:iyc+npix/2, ixc-npix/2:ixc+npix/2].mean()
        # else:
        cgray = 30000

        # L'angle de la pente
        theta = arctan2(-(yb2 - yb1), xb2 - xb1)
        # theta = arctan2( -yb2+yb1, -(xb2-xb1) )

        b1f = (xb1, yb1)
        b2f = (xb2, yb2)

    return xcenterf, ycenterf, diam, theta, b1f, b2f, cgray


def get_min_max(z, coupure=0.1):
    """
        Pour obtenir la position
        du maximum et du minimum de la derivee
        du profil de la tige z

    """

    # On vire la moyenne entre min et max
    minz = z.min()
    maxz = z.max()
    zic = z - 0.5 * (minz + maxz)
    # print z.max() - z.min()
    # figure('line')
    # clf()
    # zic = detrend( zic )
    # plot( zic)

    if (maxz - minz) < coupure:
        # print("cut ")
        ib1 = None
        ib2 = None
    else:
        # On interpole
        # zici = interp1d( arange( len(zic) ), zic )
        # Recup le gradient position du gradient max pour obtenir le bord
        gradz = gradient(sign(zic))
        # On va chercher tous les pics positifs et negatifs
        ib1 = find(gradz == 1.0)
        ib2 = find(gradz == -1.0)

        if len(ib1) > 0 and len(ib2) > 0:
            ib1 = ib1[0]
            ib2 = ib2[-1]
        else:
            ib1 = None
            ib2 = None

    return ib1, ib2



def MethodeOlivier(image, tige_table, id_tige, nbimage, xi, yi, pas, Np,
                   seuil_coupure=0.2, show_tige=False, rayonfilter=True,
                   target=None, output_fig=None):
    """
    Methode d'Olivier

    
    """

    basexi = arange(100, dtype='float32')
    # Variables
    Max_iter = tige_table.size
    # tige = Tige( id_tige, pas, size=Max_iter ) #Pour enregistrer les infos d'une tige
    cpt = 0
    bufferangle = int(3 / pas)
    passflag = True
    imp = image
    # Mon reglage avnt 0.9 et oliv 1.4
    percent_diam = 1.4

    # Astuce pour petit grain de temps dans la boucle
    add_tiges_pts = tige_table.add_point
    tdiams = tige_table.diam
    txcs = tige_table.xc
    tycs = tige_table.yc
    tthetas = tige_table.theta

    # Pour test d'un buffer sur les distance
    txb1 = tige_table.xb1
    txb2 = tige_table.xb2
    tyb1 = tige_table.yb1
    tyb2 = tige_table.yb2

    # ny, nx = shape(image)
    # fi = RectBivariateSpline(arange(nx), arange(ny), image.T, kx=1, ky=1 )
    # imp_local = local_contrast( imp, mean(xi), mean(yi) )
    # Premier transect
    xc, yc, D, theta, b1, b2, cgray = get_tige_border(xi, yi, imp, seuil_coupure=seuil_coupure)

    # plot(xni, yni,'r')
    # tige.add_point(cpt, D, xc, yc, theta, b1, b2, cgray)
    if b1 != 30000.0 and b2 != 30000.0:
        add_tiges_pts(nbimage, id_tige, cpt, D, xc, yc, theta, b1[0], b1[1], b2[0], b2[1], cgray)
        cpt += 1

    # Pour le plot en live
    if show_tige:
        if output_fig is None:
            fig = mpl.figure('test')
            axt = mpl.gca()
        else:
            print(output_fig)
            fig = output_fig
            axt = fig.get_axes()[0]
            
        # axt.imshow(image)
        linedetect, = axt.plot(xi, yi, color=(0, 1, 0), lw=2)
        b1line, = axt.plot(txb1[id_tige, :], tyb1[id_tige, :], 'co', mec='c')
        b2line, = axt.plot(txb2[id_tige, :], tyb2[id_tige, :], 'go', mec='g')
        # fig.show()
        fig.canvas.draw()
        
    # print "#########"

    # Target
    if target != None:
        xtarget = target['xc']
        ytarget = target['yc']
        rtarget = target['R']

    # Boucle jusqu'au sommet
    if xc != 30000.0 and yc != 30000.0:
        for i in xrange(Max_iter - 1):
            if show_tige:
                # mpl.figure('test')
                linedetect.set_data(xi, yi)
                b1line.set_data([txb1[id_tige, :], tyb1[id_tige, :]])
                b2line.set_data([txb2[id_tige, :], tyb2[id_tige, :]])
                # plot( [xi[0], xi[-1]], [yi[0], yi[-1]] , 'r--')
                fig.canvas.draw()

            # print theta, xc, yc
            # Angle et projection pour le tir suivant ATTENTION AU MASQUE
            # 0ld 1.4
            buffD = tdiams[id_tige, :cpt]
            if len(buffD) > bufferangle:
                RR = percent_diam * buffD[-bufferangle:].mean()
            else:
                RR = percent_diam * buffD.mean()

            # Oldway
            x1n = xc - pas * sin(theta) - RR * cos(theta)
            y1n = yc - pas * cos(theta) + RR * sin(theta)
            x2n = xc - pas * sin(theta) + RR * cos(theta)
            y2n = yc - pas * cos(theta) - RR * sin(theta)

            dx = (x2n - x1n) / float(Np - 1)
            dy = (y2n - y1n) / float(Np - 1)

            xi = basexi * dx + x1n
            yi = basexi * dy + y1n

            # imp_local = local_contrast( imp, xc, yc )
            xc, yc, D, thetat, b1, b2, cgray = get_tige_border(xi, yi, imp, seuil_coupure=seuil_coupure)

            if xc != 30000.0 and yc != 30000.0:

                # Save tige data

                add_tiges_pts(nbimage, id_tige, cpt, D, xc, yc, thetat, b1[0], b1[1], b2[0], b2[1], cgray)

                buffx = txcs[id_tige, :cpt]
                buffy = tycs[id_tige, :cpt]
                bufftheta = tthetas[id_tige, :cpt]

                if len(buffx) > bufferangle:
                    # OLD VERSION RACINE SANS THETATMP just bufferanglemean ... car bug quand entre une certaine valeur
                    # CAR singularité quand on passe de -180 a +180 (vers le bas aligné avec g !!!!) ou de +0 à -0
                    # BUG RESOLVED WITH CIRCMEAN

                    thetatmp = circmean(arctan2(-diff(buffx[-bufferangle / 2:]), -diff(buffy[-bufferangle / 2:])))
                    theta = circmean(ma.hstack([bufftheta[-bufferangle:], thetatmp]))
                    # print theta
                    tthetas[id_tige, cpt] = theta

                cpt += 1

            else:
                passflag = False

            # coupure sur le rayon si trop petit
            if rayonfilter:
                buffR = tdiams[id_tige, :cpt]
                if len(buffR) > 10:
                    Rmean = buffR[:-10].mean()
                else:
                    Rmean = None

                # Old 0.5 et 1.2
                if Rmean != None and D != 30000.0:
                    if D <= 0.5 * Rmean or D >= 1.2 * Rmean:
                        passflag = False
                        print("Interuption changement de rayon R=%0.2f moy=%0.2f" % (D, Rmean))

            if cpt >= Max_iter:
                passflag = False
                print("Iterations coupure")

            # Add a stop condition if target is defined (distance relative to target less than a value)
            if xc != None and target != None:
                dist = sqrt((xtarget - xc) ** 2 + (ytarget - yc) ** 2)
                if dist <= rtarget:
                    # print('End point reached')
                    passflag = False

            if not passflag:
                # Stop iteration
                break


###############################################################################


#implementation alternative de Queue pour avoir un qsize qui fonctionne meme sous osX

class SharedCounter(object):
    """ A synchronized shared counter.

    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.

    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/

    """

    def __init__(self, n = 0):
        self.count = mp.Value('i', n)

    def increment(self, n = 1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value

import multiprocessing.queues 

class JoinableQueue(multiprocessing.queues.JoinableQueue):
    """ A portable implementation of multiprocessing.Queue.

    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().

    """
    
    
    def __init__(self, *args, **kwargs):
        super(JoinableQueue, self).__init__(*args, **kwargs)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(JoinableQueue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(JoinableQueue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

class Queue(multiprocessing.queues.Queue):
    """ A portable implementation of multiprocessing.Queue.

    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().

    """
    
    
    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

    
class TraiteImageThread:

    def __init__(self, image_file, image_num, xypoints, max_iter,
                 pas=0.3, seuil="auto", Np=100, show_tige=False,
                 rayonfilter=False, method="Olivier", end_points={},
                 tiges_seuil_offset={}, output_fig=None):

        self.img = image_file
        self.image_num = image_num
        self.xypoints = xypoints
        self.max_iter = max_iter
        self.pas = pas
        self.seuil = seuil
        self.Np = Np
        self.show_tige = show_tige
        self.rayonfilter = rayonfilter
        self.method = method
        self.end_points = end_points
        self.tiges_seuil_offset = tiges_seuil_offset
        self.output_fig = output_fig

        # Create memory space and special variables
        self.Ntige = len(self.xypoints)
        self.tige_data = TigesNew( self.Ntige, self.max_iter )

    def __call__(self, imageprocessor):

        """
        Run the processing for the given image

        :return: The Tige data and image number
        """

        # Read the image and export them with the good
        imageprocessor.load(self.img)
        image_bw = imageprocessor.render()

        # TODO: Clean the tige_data to default value before porcessing

        for i in range(self.Ntige):
            xystart = self.xypoints[i]
            xi = linspace(xystart[0][0], xystart[1][0], self.Np, dtype='float32')
            yi = linspace(xystart[0][1], xystart[1][1], self.Np, dtype='float32')

            target = None
            if i in self.end_points:
                target = {'xc': float(self.end_points[i]['xc'][self.image_num]),
                          'yc': float(sefl.end_points[i]['yc'][self.image_num]),
                          'R': float(self.end_points[i]['R'])}

            if self.method == "Olivier":
                if self.seuil == "auto":
                    seuiln = auto_seuil(image_bw, xi, yi)
                else:
                    seuiln = self.seuil

                if i in self.tiges_seuil_offset:
                    dseuil = seuiln * (float(self.tiges_seuil_offset[i]))

                    # Quand sensibilité negative on doit augmenter le seuil pour être moins sensible au gradient d'intentensité
                    seuiln -= dseuil

                MethodeOlivier(image_bw, self.tige_data, i, self.image_num, xi, yi, self.pas, self.Np, seuiln,
                               self.show_tige, self.rayonfilter, target, self.output_fig)

        return self.image_num, self.tige_data

    def __str__(self):
        return 'Task for image %i'%self.image_num

class Worker(mp.Process):

    def __init__(self, task_queue, result_queue, image_processor):
        """

        :param task_queue: an mp.JoinableQueue with a list of collable task
        :param result_queue: an mp.Queue to store the results
        """
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.imagep = image_processor


    def run(self):
        proc_name = self.name

        # An infinit loop until we get a None in the task_queue
        while True:

            next_task = self.task_queue.get()
            if next_task is None:
                print('{} stop running: end of tasks'.format(proc_name))
                self.task_queue.task_done()
                # Stop the infinit loop
                break

            try:
                task_result = next_task(self.imagep)
            except Exception as e:
                print('Detection Failed')
                print(e)
                task_result = None
                traceback.print_exc()

            self.task_queue.task_done()
            self.result_queue.put( task_result )

def adjust_image(image, rois, gains, cut_off, disk_size):
    """
        Fonction pour aplliquer une correction de courbe (courbe sigmoid de skimage) sur certaines zones de l'image
        définies par les Region Of Interes (rois).
        Un filtre morph_open permet de diminuer les poussieres sur l'image
    """
    imgF = image

    # Les test pour savoir si on definit un int ou une list (different seuil et gain pour les differents rois)
    is_gain_list = type(gains) == type([])
    is_cutoff_list = type(cut_off) == type([])

    for i, roi in enumerate(rois):
        if is_gain_list:
            curgains = gains[i]
        else:
            curgains = gains

        if is_cutoff_list:
            curcut_off = cut_off[i]
        else:
            curcut_off = cut_off

        imgF[roi[0]:roi[1], roi[2]:roi[3]] = adjust_sigmoid(imgF[roi[0]:roi[1], roi[2]:roi[3]], curcut_off, curgains)
        # Netoyage
        if _iscv2:
            imgF[roi[0]:roi[1], roi[2]:roi[3]] = cv2.morphologyEx(imgF[roi[0]:roi[1], roi[2]:roi[3]], cv2.MORPH_OPEN,
                                                                  disk(disk_size))

    return imgF


def default_output_print(**kwargs):
    imnum = kwargs['inum']
    tot = kwargs['tot']
    print("Traitement de %i / %i" % (imnum, tot))


def ProcessImages(file_names, num_images, num_tiges, pas=0.3, seuil="auto",
                  Np=100, thread=False, show_tige=False, base_points=None,
                  rois=None, gains=20, cut_off=0.2, disk_size=4, rayonfilter=False,
                  method="Olivier", use_bw=True, color_transform=None, color_band=None,
                  output_function=default_output_print, output_function_args={}, outputdata=None,
                  end_points={}, tiges_seuil_offset={}, memory_size=10000, crops=[], output_fig=None):
    """
    Fonction pour lancer le traitement des tiges par recherche de maximum

    Arguments
    ---------

    -file_names: nom des fichier images a charger (style unix on peut utiliser *,? etc) expl: './DSC_*.JPG'
                 or a list of files ['file1.jpg','file2.jpg',etc...]
                 ou le nom d'un fichier hdf5 contenant les données 

    -num_images: nombre d'images à traiter
                  - soit un nombre
                  - soit "all" pour toutes les images
                  - soit une liste: [start, stop] avec start stop la position des images dans la liste
                                    stop = 'end' va jusqu'à la dernière image
    -num_tiges: nombre de tiges

    -pas: pas de la detection

    -seuil: seuil entre le fond de l'image et la tige.
            seuil = "auto" [defaut], "auto"
            Par defaut le seuil est calculé automatiquement pour chaque tiges à partir du 1er profil
            tracé pour définir la base des tiges

            seuil = value (entre 0 et 1)
            pour rentrer un seuil identique manuellement typiquement seuil = 0.05

    -use_bw: True -> load image in black and white, this speed up the process

    -color_transform: opencv color transformation
                      expl: cv2.COLOR_BGR2LAB

    -denoise: Apply a denoising filter to image (this may be slow !!!)

    -end_points: dict[tige_id] = {'xc':array(len(images)),'yc':array(len(images)), 'R': radius}
            Allow to stop the iterative processe when the distance between the last detected point and the end point is lest than R

    -tiges_seuil_offset: dict[tige_id] = offset seuil, en % pour ajouter au seuil auto afin de rendre + (+xx%) ou - (-xx%) sensible la detection

    -output_fig: [optional, default: None] store the figure to display the result of the treatment in live
    """

    # Ouverture des images
    # Check si c'est une liste ou une commande de recherche type unix
    if isinstance(file_names, list):
        imgs = file_names
        use_h5 = False
    elif isinstance(file_names, str) or isinstance(file_names, unicode):
        if 'h5' in os.path.splitext(file_names)[-1]:
            print(u"Images obtenues à partir du fichier h5 %s" % file_names)
            imgs = arange(0, h5store.get_number_of_images(file_names))
            use_h5 = True
        else:
            use_h5 = False
            try:
                imgs = sorted(glob.glob(file_names), key=lambda x: int(''.join(finddigits.findall(x.split('/')[-1]))))
            except:
                imgs = sorted(glob.glob(file_names))
                
    # Creation d'objet Image qui contient les specifications de transformation de l'image a traiter
    if use_h5 is None:
        image = Image(use_bw=use_bw, color_transform=color_transform, color_band=color_band)
    else:
        image = Image(use_bw=use_bw, color_transform=color_transform, color_band=color_band, h5file=file_names)
    
    # imgs = io.ImageCollection('/home/hugo/developpement/python/testgravitro/Hugo/test/manip11_07_14/apres2/*.JPG')
    if num_images == 'all':
        num_images = len(imgs)

    # Gestion des listes
    if  isinstance(num_images, list) and len(num_images) == 2:
        img_start = num_images[0]
        img_stop = num_images[1]

        if img_stop == 'end':
            img_stop = len(imgs)

        imgs = imgs[img_start:img_stop]
        print("traitement de %s -> %s" % (imgs[0], imgs[-1]))
        num_images = len(imgs)

    Num_tiges = num_tiges
    xypoints = base_points
    # print(imgs, Num_tiges, xypoints)
    ##############

    # Lancement du traitement avec ou sans threads
    ta = time.time()
    if _windows:
        results = mp.Queue()
        tasks = mp.JoinableQueue()
    else:
        results = Queue() # to store output of workers
        tasks = JoinableQueue()

    num_worker = mp.cpu_count()  # Nombre de processeurs
    num_images = len(imgs)

    # Pas trop de worker si pas beaucoup d'images
    if num_worker > num_images:
        num_worker = num_images
        
    output_function_args['old_inum'] = 0
    output_function_args['inum'] = 0
    output_function_args['tot'] = num_images

    # Create workers
    workers = [ Worker(tasks, results, image) for i in range(num_worker) ]

    # Start their infinite loop in process
    for w in workers:
        w.start()

    # Add callable class to our task list
    for i, img_name in enumerate(imgs):
        tasks.put( TraiteImageThread( img_name, i, xypoints, memory_size, pas, seuil, Np, show_tige, rayonfilter,
                                      method, end_points, tiges_seuil_offset, output_fig) )

    # Add the stop to kill workers at the end
    for i in range(num_worker):
        tasks.put( None )

    # Loop to display information
    while tasks.qsize() > 0:
        time.sleep(0.2)
        inum = results.qsize()
        output_function_args['inum'] = inum
        output_function_args['tot'] = num_images
        output_function(**output_function_args)
        output_function_args['old_inum'] = inum

    # Wait untill all task have been done
    tasks.join()

    output_data = TigesManager(len(xypoints), num_images, memory_size)

    # Retrive all_data
    all_results = {}
    for i in range(len(imgs)):
        img_num, tigedata = results.get()
        all_results[img_num] = tigedata

        for attr in ['diam', 'xc', 'yc', 'theta', 'xb1', 'yb1', 'xb2', 'yb2']:
            getattr(output_data, attr)[:,img_num,:] = getattr(tigedata, attr)

    print("Done in %f s" % (time.time() - ta))

    print('Compress data')
    output_data.compress_data()
    infos = [{'imgname': iname, 'iimg': ii} for ii, iname in enumerate(imgs)]

    output = {'tiges_data': output_data, 'tiges_info': infos}

    print('Done')
    if outputdata is not None:
        outputdata.put({"data": output, "imgs": imgs, "xypoints": xypoints})
        return
    else:
        return output, imgs, xypoints

###############################################################################
#  Fonction diverses
###############################################################################


def auto_seuil(image, xi, yi):
    """Fonction pour determiner automatiquement le seuil a partir du premier profil"""

    # ligne de gris
    if _iscv2:
        zi = cv2.remap(image, xi, yi, cv2.INTER_LINEAR)[:, 0]
    else:
        zi = ndimage.map_coordinates(image.T, vstack((xi, yi)), order=1)

    # print zi.max(), zi.min(), zi.max()-zi.min()
    # Condition si il y a une tige ou pas
    if (zi.max() - zi.min()) < 0.05:
        # pas de tige ont met le seuil a 1 ce qui coupe la detection
        seuil = 1.0
    else:
        seuil = (zi.max() - zi.min()) * 0.3

    return seuil


def save_results(results, nameout):
    # Dump
    with open(nameout, "wb") as out:
        pickle.dump(results, out, protocol=2)


def load_results(file_name):
    with open(file_name, 'rb') as fin:
        data = pickle.load(fin)

    return data


###############################################################################
#                     TRAITEMENT DES TIGES EXTRAITES                          #
###############################################################################
def moving_avg(x, W):
    W = int(W)
    # La lenteur
    # S = array( [ mean( x[i-W/2:i+W/2] ) for i in arange(W/2, len(x)-W/2)] )
    xT = ma.hstack([x[W / 2 - 1:0:-1], x, x[-1:-W / 2 - 1:-1]])
    wind = ones(W)

    # Comptibilitée version de numpy
    try:
        S = ma.convolve(wind / wind.sum(), xT, mode='valid')
    except:
        S = convolve(wind / wind.sum(), xT, mode='valid')

    return S


def traite_tige2(xc, yc, R, pas, coupe=5):
    """
        Fonction pour traiter une seul tige

        tige
        ----
        Dictionnaire contenant les infos d'une tige (cf fonction tige2dict pour convertir la class tige en dict)

        coupe
        -----
        enleve x points à la fin de la detection de la tiger

    """

    xc = xc[~xc.mask]  # prend que les donnees valides
    yc = yc[~yc.mask]
    R = R[~R.mask]

    # base de la tige a zero
    # print(xc)
    x = xc - xc[0]
    y = yc - yc[0]

    # Taille de la tige
    Tt = sqrt(x[-1] ** 2 + y[-1] ** 2)

    # Filtre les tiges trop petites
    if Tt > 1.5 * R.mean() and len(x) > coupe * 2:
        # Retire coupe point au bout
        x = x[:-coupe]
        y = y[:-coupe]
        R = R[:-coupe]

        # Retourne y car photo axes dans autre sens
        y = -y

        # Rayon moyen pour estimer la taille du smooth
        N = round(R.mean() / pas)
        sx = moving_avg(x, N)
        sy = moving_avg(y, N)

        # Add zero at the begining (because moving avg remove wind/2 pts at stat and wind/2 pts at the end)
        # sx = hstack((x[0], sx))
        # sy = hstack((y[0], sy))

        dx, dy = diff(sx), diff(sy)

        ds = sqrt(dx ** 2 + dy ** 2)
        s = cumsum(ds)

        T = arctan2(-dx, dy)



    else:
        sx, sy, T, s, N = None, None, None, None, None

    return sx, sy, T, s, N


def traite_tiges2(tiges, itige=None, pas=0.3, causette=True, return_estimation_zone=False):
    """
        Fonction pour traiter les tiges detectées pour l'expérience du gravitron
        **return allx, ally, L, thetab, lines**


        tiges
        -----
        Liste des tiges (au format dictionnaire) en fonction du temps obtenus par la fonction load_results
        Tiges, data = load_results( 'file' )

        itiges
        ------
        Si *None* (option par default) prend toute les tiges, sinon doit etre une liste contenant les indices des tiges a traiter

        return_estimation_zone
        ----------------------
        Sort la zone sur laquelle on fait la moyenne
    """

    if not itige:
        Ntige = len(tiges.xc)
        itige = range(Ntige)
    else:
        Ntige = len(itige)

    lines = [None] * Ntige

    bad_value = -30000
    Ntime = len(tiges.xc[0, :, 0])
    Max_tige_lengh = len(tiges.xc[0, 0, :])  # Taille de la convolution possible
    allx = ma.masked_equal(zeros((Ntige, Ntime, Max_tige_lengh)) + bad_value, bad_value)
    ally = ma.masked_equal(zeros((Ntige, Ntime, Max_tige_lengh)) + bad_value, bad_value) 
    L = ma.masked_equal(zeros((Ntige, Ntime)) + bad_value, bad_value)
    thetab = ma.masked_equal(zeros((Ntige, Ntime)) + bad_value, bad_value)
    measure_zone = []  # List to store indices where we do the mean of values

    # boucle sur les tiges
    for i in xrange(Ntige):
        linest = []
        measure_zonet = []
        # boucle sur le temps
        for t in xrange(Ntime):
            # Test si pas de detection des le debut
            tmp = ma.masked_invalid(tiges.xc[itige[i], t, :])
            if len(tmp[~tmp.mask]) > 1:
                xt, yt, theta, s, N = traite_tige2(tiges.xc[itige[i], t, :], tiges.yc[itige[i], t, :],
                                                   tiges.diam[itige[i], t, :] / 2., pas)
                if xt is None and yt is None:
                    if causette:
                        print("La taille de la tige %i sur l'image %i est trop faible" % (itige[i], t))
                        
                    measure_zonet += [(0, 0)] # sinon on a des inconsistance de taille 
                else:
                    di = round(1.5 * N)
                    measure_zonet += [(-(N + di), -N)]
                    # x0,y0 = tiges.xc[itige[i],t,~tiges.xc[itige[i],t].mask][0], tiges.yc[itige[i],t,~tiges.yc[itige[i],t].mask][0]
                    #Ancienne version
                    #linest += [zip(xt, yt)]

                    # thetab[ i, t ] = ma.mean(theta[-(N+di):-N])
                    # Reference 0 quand tige est verticale
                    thetab[i, t] = rad2deg(circmean(theta[-int(N + di):-int(N)], high=pi, low=-pi))

                    L[i, t] = s[-1]
                    # lines[ i, t, :len(xt) ] = zip(xt, yt)
                    allx[i, t, :len(xt)] = xt
                    ally[i, t, :len(yt)] = yt

            else:
                if causette:
                    print(u"La tige %i sur image %i a merdée" % (itige[i], t))
                measure_zonet += [(0, 0)] # sinon on a des inconsistance de taille 

        #lines[i] = linest
        #version plus rapide
        lines[i] = ma.dstack((allx[i], ally[i]))
        measure_zone += [measure_zonet]

    if return_estimation_zone:
        return allx, ally, L, thetab, lines, measure_zone
    else:
        return allx, ally, L, thetab, lines

def get_tiges_lines(xc, yc):
    """
    Fonction qui retourne un liste pour chaque tiges une liste de
    zip(xc,yc) compatible pour tracer un linecollection avec
    matplotlib

    Paramètres:
    -----------

    xc, array (dim: Ntiges x Ntimes x abscisse curv)
    yc, array (dim: Ntiges x Ntimes x abscisse curv)
    """

    Ntige = len(xc)
    Ntime = len(xc[0, :, 0])
    lines = [None] * Ntige
    # boucle sur les tiges
    for i in xrange(Ntige):
        """
        linest = []
        # boucle sur le temps
        for t in xrange(Ntime):
            linest += [zip(xc[i, t], yc[i, t])]
        lines[i] = linest
        """
        # Version plus rapide
        lines[i] = ma.dstack((xc[i], yc[i]))
        
    return lines


def traite_tiges_thread(tiges, itige=None, pas=0.3, causette=True, return_estimation_zone=False):
    """
    Function to use multithread treatment of tiges data
    """

    pass


def plot_tiges(xt, yt, Lt, Tt, linest):
    from mpl_toolkits.axes_grid1 import AxesGrid
    from matplotlib.collections import LineCollection

    fig = mpl.figure('tiges')

    Ntige = len(xt)
    G = mpl.GridSpec(3, Ntige)
    grid2 = AxesGrid(fig, G[0, :],  # similar to subplot(142)
                     nrows_ncols=(1, Ntige),
                     axes_pad=0.0,
                     share_all=True,
                     label_mode="1",
                     cbar_location="top",
                     cbar_mode="single",
                     cbar_size="0.5%",
                     aspect=False,
                     cbar_pad=0.02
                     )

    maxX, maxY = 0, 0
    minX, minY = 0, 0
    cmin = -Tt.mean(0).max()
    cmax = Tt.mean(0).max()

    for it in xrange(Ntige):
        lcollec = LineCollection(linest[it], cmap=mpl.cm.PiYG_r, linewidth=(2,), alpha=1,
                                 norm=mpl.Normalize(vmin=cmin, vmax=cmax))

        lcollec.set_array(Tt[it])

        grid2[it].add_collection(lcollec)

    maxX, minX = xt.max(), xt.min()
    maxY, minY = yt.max(), yt.min()

    grid2.cbar_axes[0].colorbar(lcollec)
    grid2.axes_llc.set_xlim([minX, maxX])
    grid2.axes_llc.set_ylim([minY, maxY])
    grid2.axes_llc.set_xticks([int(minX), 0, int(maxX)])

    # Les moyennes
    axa1 = mpl.subplot(G[1, :])
    mpl.plot(Lt.T)
    mpl.plot(Lt.mean(0), 'k', lw=3)
    mpl.legend([str(i) for i in xrange(Ntige)], loc=0, ncol=3, title='tiges',
               framealpha=0.3)
    mpl.subplot(G[2, :], sharex=axa1)
    mpl.plot(Tt.T)
    mpl.plot(Tt.mean(0), 'k', lw=3)


def extract_Angle_Length(data_path, output_file_name, global_tige_id=None,
                         image_path_mod=lambda pname: pname['imgname'], methode_de_traitements=traite_tiges2, xy=False,
                         get_time_from_photos=True):
    """
    Function to extract only the Tip Angle, the length and the pictures time.

    It create a pandas DataFrame with
    'tige', 'angle', 'rayon','temps', 'taille', 'sequence', 'angle_0_360'

    if option xy = True add 'xy'->(xi,yi) for each stems to the table

    Inputs
    ======

    data_path: path to the pickle file from the raw treatment, can be a list in chronological order
    output_file_name: name of file to save

    Options
    =======

    image_path_mod: function to change the path of pictures from data['tiges_info']['imgname']
                    The default function return the path contain in data['tiges_info']['imgname'] without modification

                    If data_path is a list, this also need to be a list with the same size.


    methode_de_traitements: function to process data['tiges_data'] to get Angle and Length

    global_tige_id: Function to map the id of tige to make a coerent global id
                    boite du haut [1 2 3]
                    boite du milieu [3,4,5]
                    boite du bas [5,6,7]

                    Exemple: format to map id (0->1) (1->5) for sequence 0 and (0->5) for sequence 2:
                    [ {0:1, 1:5}, {0:5} ]

    get_photo_time: Option to extract time from EXIF infos on images

    Output
    ======

    csv with:
        - 'tige', 'angle', 'temps', 'taille', 'sequence', 'angle_0_360'
    """

    import pandas as pd

    # Manage type and convert inputs to list
    if type(data_path) != type([]):
        data_path = [data_path]
    if type(image_path_mod) != type([]):
        image_path_mod = [image_path_mod]

        if len(image_path_mod) != len(data_path):
            image_path_mod = [image_path_mod[0]] * len(data_path)

    # Global data
    if xy:
        data_out = pd.DataFrame(
            columns=['tige', 'angle', 'temps', 'taille', 'rayon', 'x', 'y', 'nom photo', 'sequence', 'angle_0_360'])
    else:
        data_out = pd.DataFrame(columns=['tige', 'angle', 'temps', 'taille', 'rayon', 'sequence', 'angle_0_360'])

    # loop over list (sequence)
    for i in range(len(data_path)):
        # Here is to get ride of windows path accent bugs
        try:
            print(u"Proceed %s" % data_path[i].decode('utf8'))
        except:
            print("Proceed data")
        # Load data
        data_in = load_results(data_path[i])
        tiges = data_in['tiges_data']
        R = tiges.diam / 2.0
        xt, yt, ll, aa, _ = methode_de_traitements(tiges)
        # Le temps a partir des photos
        if get_time_from_photos:
            try:
                t = [get_photo_time(image_path_mod[i](a)) for a in data_in['tiges_info']]
            except:
                print(u"No exif info on picturs to get time")
                t = arange(0, len(data_in['tiges_info']))
        else:
            t = arange(0, len(data_in['tiges_info']))

        # Loop over tige
        Ntiges = len(aa)
        for n in range(Ntiges):
            # Number of time records
            nstep = len(aa[n])

            # map tige id
            if global_tige_id != None:
                if n in global_tige_id[i]:
                    tigetmp = [global_tige_id[i][n]] * nstep
                else:
                    tigetmp = [n + 1] * nstep
            else:
                tigetmp = [n + 1] * nstep

            if xy:
                data_tmp = pd.DataFrame({'tige': tigetmp, 'angle': aa[n],
                                         'temps': t, 'taille': ll[n],
                                         'rayon': R[n].mean(1),
                                         'x': xt[n].tolist(),
                                         'y': yt[n].tolist(),
                                         'sequence': [i] * nstep,
                                         'angle_0_360': convert_angle(aa[n]),
                                         'nom photo': [os.path.basename(pnames['imgname']) for pnames in
                                                       data_in['tiges_info']]})
            else:
                data_tmp = pd.DataFrame({'tige': tigetmp, 'angle': aa[n],
                                         'temps': t, 'taille': ll[n],
                                         'rayon': R[n].mean(1),
                                         'sequence': [i] * nstep,
                                         'angle_0_360': convert_angle(aa[n])})

            data_out = data_out.append(data_tmp, ignore_index=True)

    print(data_out.head())

    # Save data to a csv
    if output_file_name != None:
        print(u"Saved to %s" % output_file_name)
        data_out.to_csv(output_file_name, index=False)       
    else:
        return data_out


def convert_angle(angle):
    # Convert angle from 0->180 and -0 -> -180 to 0->360
    return angle % 360


def plot_sequence(data, tige_color=None, show_lims=True,
                  ydata='angle', tige_alpha=0.5):
    """
        Function to quickly plot sequence from pandas table

        tige_color -> dict[tige_id] = "color"
    """

    Nsec = data.sequence.unique()
    Ntige = data.tige.unique()

    # Automatic color
    if tige_color is None:
        tige_color = mpl.cm.Set1(linspace(0, 1, len(Ntige) + 1))

    for i, tige in enumerate(Ntige):
        for sec in Nsec:
            # La couleur il faut enlever le alpha
            tc = tige_color[i]
                
            # Try except because number of tige may change for each sequence
            try:
                dataq = data.query('sequence == %i and tige == "%s"' % (sec, tige))
                dataq.plot(x='temps', y=ydata, ax=mpl.gca(),
                           color=[tc], alpha=tige_alpha, legend=False)
            except Exception as e:
                print("Impossible de tracer la tige %s" % tige)
                print(e)
                traceback.print_exc()

    # Plot the mean over
    ax = mpl.gca()
    for sec in Nsec:
        y_mean = data[data.sequence == sec].groupby('temps').mean()[ydata]
        ax.plot(data[data.sequence == sec].temps.unique(), y_mean, color='RoyalBlue', lw=2)

        if show_lims:
            # Plot sec limites
            yminmax = mpl.ylim()
            tmin = data[data.sequence == sec].temps.min()
            tmax = data[data.sequence == sec].temps.max()

            ax.plot([tmin, tmin], yminmax, '--', color='gray')
            ax.plot([tmax, tmax], yminmax, '--', color='gray')

    # ylabel
    mpl.ylabel(ydata)


###############################################################################


######################### TRACK PATTERN TO MAKE A STOP POINT ##################
def find_pattern(img, pattern, xy_pattern_center, max_box_shift=1, debug=False):
    # La taille du pattern
    pheight, pwidth = pattern.shape
    iheight, iwidth = img.shape
    # Point median
    p_xc, p_yc = xy_pattern_center

    # Borne de la zone d'exploration
    dx = max_box_shift * pwidth
    dy = max_box_shift * pheight
    good_d = max((dx, dy))

    binfx, bsupx = p_xc - good_d, p_xc + good_d
    binfy, bsupy = p_yc - good_d, p_yc + good_d

    # print(binfx,bsupx, binfy,bsupy)
    if binfx < 0:
        binfx = 0
    if binfy < 0:
        binfy = 0

    if bsupx >= iwidth:
        bsupx = iwidth - 1
    if bsupy >= iheight:
        bsupy = iheight - 1

    binfx, bsupx = int(binfx), int(bsupx)
    binfy, bsupy = int(binfy), int(bsupy)

    # Crop the image
    imgc = img[binfy:bsupy, binfx:bsupx].copy()

    if debug:
        mpl.figure('cropped img')
        mpl.imshow(imgc)

    # Find the pattern
    # patternc = cv2.blur(pattern, (5,5))
    # imgcc = cv2.blur(imgc, (5,5))
    sizeloc = good_d / 4
    if sizeloc % 2 == 0:
        sizeloc += 1

    if sizeloc < 5:
        sizeloc = 5

    patternb = cv2.adaptiveThreshold(pattern, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, sizeloc, -3)
    imgcb = cv2.adaptiveThreshold(imgc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, sizeloc, -3)

    corr1 = signal.fftconvolve(imgcb, patternb[::-1, ::-1], mode='same')
    # corr2 = signal.fftconvolve(imgcc, patternc[::-1,::-1], mode='same')
    # y_raw, x_raw = unravel_index(np.argmax(corr), corr.shape)
    y_raw1, x_raw1 = where(corr1 == corr1.max())
    # y_raw2, x_raw2 = where(corr2==corr2.max())
    x_raw = x_raw1[0]
    y_raw = y_raw1[0]

    if debug:
        print(good_d)
        print(binfx, bsupx, binfy, bsupy)
        print(sizeloc)
        print(x_raw, y_raw)
    # res = cv2.matchTemplate(imgcb, patternb, cv2.TM_CCORR_NORMED)
    # vmin, vmax, imin, imax = cv2.minMaxLoc( res )
    # y_raw, x_raw = imax

    # Nouveau point de l'opposer du rectangle et celui du centre
    # y_nend, x_nend = y_raw+pheight, x_raw+pwidth
    # xnm, ynm = mean((x_raw,x_nend)), mean((y_raw,y_nend))
    xstart, xend = int(x_raw - pwidth / 2), int(x_raw + pwidth / 2)
    ystart, yend = int(y_raw - pheight / 2), int(y_raw + pheight / 2)
    xnm, ynm = 0.5 * (xstart + xend), 0.5 * (ystart + yend)
    new_pattern = imgc[int(ystart):int(yend) + 1, int(xstart):int(xend) + 1].copy()

    if debug:
        mpl.figure('pattern canny')
        mpl.cla()
        mpl.subplot(131)
        mpl.imshow(patternb)
        mpl.subplot(132)
        mpl.imshow(imgcb)
        mpl.plot((xstart, xend, xend, xstart, xstart), (ystart, ystart, yend, yend, ystart), 'r--')
        mpl.plot(xnm, ynm, 'mo')
        mpl.plot(x_raw, y_raw, 'wo')
        mpl.subplot(133)
        mpl.imshow(corr1)

        mpl.figure('cropped img')
        mpl.plot(xnm, ynm, 'mo')
        mpl.figure('new_pattern')
        mpl.imshow(new_pattern)

    # Remet les points dans les coordonnees generale de img
    return xnm + binfx, ynm + binfy, new_pattern


def compute_pattern_motion(images_names, pattern, pattern_center, output_data=None, output_function=None):
    if _iscv2:
        xcenter_tot = zeros_like(images_names)
        ycenter_tot = zeros_like(xcenter_tot)

        # Loop over images name
        xp, yp = pattern_center
        imgtmp = Image(use_bw=True)
        output_function_args = {}
        num_images = len(images_names)
        output_function_args['tot'] = num_images
        output_function_args['old_inum'] = 0

        for i, name in enumerate(images_names):
            # Load the image
            imgtmp.load(name)
            xp, yp, pattern = find_pattern(imgtmp.render(rescale=False), pattern, (xp, yp))
            xcenter_tot[i] = xp
            ycenter_tot[i] = yp

            if output_function == None:
                print("traitement de %i/%i" % (i, len(images_names)))
            else:
                output_function_args['inum'] = i
                output_function(**output_function_args)
                output_function_args['old_inum'] = i

        if output_data == None:
            return xcenter_tot, ycenter_tot
        else:
            output_data.put({"xc": xcenter_tot, "yc": ycenter_tot})
    else:
        print('You need to install python-opencv')
