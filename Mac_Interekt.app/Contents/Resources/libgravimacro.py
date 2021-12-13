# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Portagee de la detection de tige de Olivier (en matlab) vers python

Regroupe les differentes fonctions pour le traitements des images de la manip
du gravitron. 

Version: 2.0.2
Date: 14/12/2015

Modif
-----
15/12/2015 Hugo: Changement du seuil de l'auto detection de 0.1 à 0.05 TODO: Passer ce seuil en option modifiable!
06/10/2015 Hugo: Ajout d'un seuil dans auto_seuil pour voir si il y a vraiment une tige... permet de traiter des séries ou les tiges
                 sont inexistantes au début.
                 
01/09/2015 Hugo: Grosse correction de la moyenne des angles, utilise circmean (permet de ne pas avoir moyenne a zeros qd +/-180 deg.
                 Marche pour mieux surout pour les racines qui passent par -180/180 !
                 
30/07/2015 Hugo: Correction bug angle pour les racines dans traite_tige2, il ne faut pas inverse y 
25/06/2015 Hugo: Ajout d'option dans Extract_AngleTaille pour ne pas sortir de fichier mais retourner le tableau pandas
12/06/2015 Hugo: Correction d'un bug dans la moyenne des theta qui causait des inversion de direction
12/05/2015 Hugo: Ajout fonction pour faire le tableau propre avec pandas des sequences
                 extract_Angle_Lenght
                 Fonction pour tracer les sequences et les joindes pour le transitoire
                 
18/02/2015 Hugo: Ajout d'une nouvelle methode basée sur la detection de contour 

30/01/2015 Hugo: Changement de version car netoyage des methodes qui ne servent plus (cercle, croix)
                 Plus de classe Tige, tout est géré par la Classe Tiges 
                 Gestion d'un array special pour les thread (gros gains en mémoire et en vitesse!)
                 Seuil determiné automatiquement sur le premier trait
                 
13/01/2014 Hugo:
    Ajout d'un debut d'aide pour la fonction principale de traitement
    Ajout option pour donner une plage d'image a traiter num_images = [start, stop]
    
04/12/2014 Hugo:
- Ajout du param gain et seuil pour les zones de contraste

16/10/2014 hugo:
- Ajout des zones d'interet avec augmentation du contraste
- petite correction de bugs dans la methode d'Olivier

24/09/14 Hugo:
-Utilisation de cv2.remap a la place de ndimage.map_coordinate petit gain en vitesse
-affichage progression avec les thread

22/09/14 Hugo:
-ajout du filtre proche du centre dans get_tige_border
-modification de traite_tiges et plot_tiges

18/09/14 Hugo:
-Changement de la sauvegarde et de la classe Tige
-Ajout des fonctions de traitement

05/09/14
-Premiere version
"""


#Imports des bibliothéques
from pylab import * 

try:
    #Pour charger les image rapidement
    import cv2 
    _iscv2 = True
except:
    _iscv2 = False
    
import glob
import cPickle as pickle
#from scipy.signal import argrelmax
from scipy.stats import circmean #Pour faire des moyenne d'angle correcte!!!
from scipy.optimize import fmin
from scipy import ndimage
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

#Pour la methode Hugo

#Pour extraire le temps des photos
try:
    from PIL import Image as pilimage
except:
    import Image as pilimage
    
import datetime
import re


finddigits = re.compile(r'\d+?')

      
class Image():
    
    """
        Class pour gerer la façon dont on ouvre les image, utiliser la librarie cv2
        et peut appliquer toutes les conversions de couleurs contenues dans cv2.COLOR_
        
        Expl avec param par defaut
        Image_reader().load( name ) 
        
        Option
            
            band
            
            color_converter: 
            
            cv2.COLOR_BGR2LAB
            

    """
    
    def __init__(self, fname=None, use_bw=False, color_transform=None, color_band=None):
    
        self.data = None
        self.use_bw = use_bw
        self.color_transform = color_transform
        self.color_band = color_band
        
        #Check if cv2 is loaded to open images
        if _iscv2:
            self.loader = cv2.imread
        else:
            self.loader = pilimage.open
            
        if fname != None:
            self.load(fname, use_bw)
        
        
    
    def load(self, fname):
        if self.use_bw:
            if _iscv2:
                self.data = self.loader(fname, 0)
            else:
                self.data = array(self.loader(fname).convert('L'))
        else:
            self.data = array(self.loader(fname))
            
    def render(self):
        
        if _iscv2:
            #Check if we need to transform color
            if self.color_transform != None:
                out = cv2.cvtColor( self.data, self.color_transform )
            else:
                out = self.data
                
            #Check if we have select a specific image band
            if self.color_band != None:
                out = out[:,:,self.color_band]        
        else:
            out = self.data
            
            
        return out/255.
        
    def __repr__(self):
        return self.render()




class Tiges():
        
    def __init__(self, nbtige, nbimage, size=2000, thread_safe=False):
        """
            Class pour stocker toutes les tiges dans un tableau nbimage x nbtige x size
        """

        if thread_safe:
            typed = ctypes.c_float
            shared_diam = mp.Array(typed, nbtige*nbimage*size)
            shared_xc = mp.Array(typed, nbtige*nbimage*size)
            shared_yc = mp.Array(typed, nbtige*nbimage*size)
            shared_theta = mp.Array(typed, nbtige*nbimage*size)
            shared_xb1 = mp.Array(typed, nbtige*nbimage*size)
            shared_yb1 = mp.Array(typed, nbtige*nbimage*size)
            shared_xb2 = mp.Array(typed, nbtige*nbimage*size)
            shared_yb2 = mp.Array(typed, nbtige*nbimage*size)
            shared_gray_level = mp.Array(typed, nbtige*nbimage*size)
            
            self.diam = ctypeslib.as_array(shared_diam.get_obj()).reshape(nbtige, nbimage, size)
            self.xc = ctypeslib.as_array(shared_xc.get_obj()).reshape(nbtige, nbimage, size)
            self.yc = ctypeslib.as_array(shared_yc.get_obj()).reshape(nbtige, nbimage, size)
            self.theta = ctypeslib.as_array(shared_theta.get_obj()).reshape(nbtige, nbimage, size)
            self.xb1 = ctypeslib.as_array(shared_xb1.get_obj()).reshape(nbtige, nbimage, size)
            self.yb1 = ctypeslib.as_array(shared_yb1.get_obj()).reshape(nbtige, nbimage, size)
            self.xb2 = ctypeslib.as_array(shared_xb2.get_obj()).reshape(nbtige, nbimage, size)
            self.yb2 = ctypeslib.as_array(shared_yb2.get_obj()).reshape(nbtige, nbimage, size)
            self.gray_level = ctypeslib.as_array(shared_gray_level.get_obj()).reshape(nbtige, nbimage, size)
            
            self.diam.fill(30000)
            self.xc.fill(30000)
            self.yc.fill(30000)
            self.theta.fill(30000)
            self.xb1.fill(30000)
            self.xb2.fill(30000)
            self.yb1.fill(30000)
            self.yb2.fill(30000)
            self.gray_level.fill(30000)
            
        else:
            modeltab = ones( [nbtige, nbimage, size] ) * 30000
                
            self.diam =  ma.masked_equal( modeltab, 30000 )
            self.xc =  ma.masked_equal( modeltab, 30000 )
            self.yc = ma.masked_equal( modeltab, 30000 )
            self.theta = ma.masked_equal( modeltab , 30000 )
            self.xb1 = ma.masked_equal( modeltab, 30000 )
            self.yb1 = ma.masked_equal( modeltab, 30000 )
            self.xb2 = ma.masked_equal( modeltab, 30000 )
            self.yb2 = ma.masked_equal( modeltab, 30000 )
            self.gray_level = ma.masked_equal( modeltab, 30000 )

    def add_point(self,nbimage,id_tige,pos_in_tige,diam, xc, yc, theta, xb1, yb1, xb2, yb2, graylevel):
        """
            Add values to the structure
        """
        
        self.diam[id_tige, nbimage, pos_in_tige ] = diam  
        self.xc[id_tige, nbimage, pos_in_tige ]  =  xc
        self.yc[id_tige, nbimage, pos_in_tige ]  = yc
        self.theta[id_tige, nbimage, pos_in_tige ]  = theta
        self.xb1[id_tige, nbimage, pos_in_tige ]  = xb1
        self.yb1[id_tige, nbimage, pos_in_tige ]  = yb1
        self.xb2[id_tige, nbimage, pos_in_tige ]  = xb2
        self.yb2[id_tige, nbimage, pos_in_tige ]  = yb2
        self.gray_level[id_tige, nbimage, pos_in_tige ]  = graylevel 
        
    def Mask_invalid(self):
        
        self.diam =  ma.masked_equal( self.diam, 30000 )
        self.xc =  ma.masked_equal( self.xc, 30000 )
        self.yc = ma.masked_equal( self.yc, 30000 )
        self.theta = ma.masked_equal( self.theta , 30000 )
        self.xb1 = ma.masked_equal( self.xb1, 30000 )
        self.yb1 = ma.masked_equal( self.yb1, 30000 )
        self.xb2 = ma.masked_equal( self.xb2, 30000 )
        self.yb2 = ma.masked_equal( self.yb2, 30000 )
        self.gray_level = ma.masked_equal( self.gray_level, 30000 )
        
        

def get_tige_border(xi, yi, image, seuil_coupure=0.1):
    """
        Obtenir les bords de la tige a partir du profil
        
    """
    

    #zi = ndimage.map_coordinates(image.T, vstack( (xi, yi) ), order=1 )
    #cv2.INTER_LINEAR
    if _iscv2:
        zi = cv2.remap(image, xi, yi, cv2.INTER_LINEAR)[:,0]
    else:
        zi = ndimage.map_coordinates(image.T, vstack( (xi, yi) ), order=1 )
    
    #figure('line')
    #plot( zi )
    ib1, ib2 = get_min_max( zi, coupure=seuil_coupure )
    if type(ib1) == type(None) or type(ib2) == type(None):
        xcenterf = None
        ycenterf = None
        diam = None
        theta = None
        b1f = None
        b2f = None
        cgray = None
    else: 
            
        #Calcul du centre et du rayon
        xb1, yb1 = xi[ib1], yi[ib1]
        xb2, yb2 = xi[ib2], yi[ib2]

        xcenterf, ycenterf = 0.5 * ( xb1 + xb2 ), 0.5 * ( yb1 + yb2 )
        diam = sqrt( (xb2 - xb1) **2 + (yb1 - yb2)**2 )
        
        #Get the level of gray at center
        ixc, iyc = int(xcenterf), int(ycenterf)
        npix = int(diam/2) #taille du rectangle pour faire la moyenne sur les pixels
        cgray = image[iyc-npix/2:iyc+npix/2, ixc-npix/2:ixc+npix/2].mean()
        
        
        #L'angle de la pente 
        theta = arctan2( -(yb2-yb1), xb2-xb1 )
        #theta = arctan2( -yb2+yb1, -(xb2-xb1) )
        
        b1f = (xb1, yb1)
        b2f = (xb2, yb2)
            
        
    return xcenterf, ycenterf, diam, theta, b1f, b2f, cgray
    
def get_min_max( z, coupure=0.1):
    """
        Pour obtenir la position 
        du maximum et du minimum de la derivee 
        du profil de la tige z

    """
    
     #On vire la moyenne entre min et max
    minz = z.min()
    maxz = z.max()
    zic =  z - 0.5 * ( minz + maxz )
    #print z.max() - z.min()
    #figure('line')
    #clf()
    #zic = detrend( zic )
    #plot( zic)
    
    if ( maxz - minz ) < coupure:
        #print("cut ")
        ib1 = None
        ib2 = None
    else:
        #On interpole
        #zici = interp1d( arange( len(zic) ), zic )
        #Recup le gradient position du gradient max pour obtenir le bord
        gradz = gradient( sign(zic) )
        #On va chercher tous les pics positifs et negatifs
        ib1 = find( gradz == 1.0 )
        ib2 = find( gradz == -1.0 )
        
        if len(ib1) > 0 and len(ib2) > 0:
            ib1 = ib1[0]
            ib2 = ib2[-1]
        else:
            ib1 = None
            ib2 = None

            
    return ib1, ib2

    
def methode_Olivier(image, tiges_table, id_tige, nbimage, xi, yi, pas, Np, seuil_coupure=0.2, show_tige = False, rayonfilter=True):
    """
        Methode d'Olivier
    """
    
    basexi = arange(100, dtype='float32')    
    #Variables 
    Max_iter = 2000
    #tige = Tige( id_tige, pas, size=Max_iter ) #Pour enregistrer les infos d'une tige
    cpt = 0
    bufferangle = int(3/pas)
    passflag = True
    imp = image
    
    #Astuce pour petit grain de temps dans la boucle 
    add_tiges_pts = tiges_table.add_point 
    tdiams = tiges_table.diam
    txcs = tiges_table.xc
    tycs = tiges_table.yc
    tthetas = tiges_table.theta
    
    #Pour test d'un buffer sur les distance
    txb1 =  tiges_table.xb1
    txb2 =  tiges_table.xb2
    tyb1 = tiges_table.yb1
    tyb2 = tiges_table.yb2
    
    
    #ny, nx = shape(image)
    #fi = RectBivariateSpline(arange(nx), arange(ny), image.T, kx=1, ky=1 )
    #imp_local = local_contrast( imp, mean(xi), mean(yi) )
    #Premier transect 
    xc, yc, D, theta, b1, b2, cgray = get_tige_border(xi, yi, imp, seuil_coupure=seuil_coupure)
    
    #plot(xni, yni,'r')
    #tige.add_point(cpt, D, xc, yc, theta, b1, b2, cgray)
    if b1 and b2:
        add_tiges_pts(nbimage,id_tige, cpt, D, xc, yc, theta, b1[0], b1[1], b2[0], b2[1], cgray )    
        cpt += 1
    
    #Pour le plot en live
    if show_tige:
        figure('test')
        linedetect, = plot(xi, yi , color=(0,1,0) , lw=2) 
        b1line, = plot( txb1[id_tige, nbimage, :], tyb1[id_tige, nbimage, :], 'co', mec='c')
        b2line, = plot(txb2[id_tige, nbimage, :], tyb2[id_tige, nbimage, :], 'go', mec='g')
        draw()
    #print "#########"
        

    #Boucle jusqu'au sommet
    if xc and yc:
        for i in xrange(Max_iter-1):
            if show_tige:
                figure('test')
                linedetect.set_data( xi, yi )
                b1line.set_data( [txb1[id_tige, nbimage, :], tyb1[id_tige, nbimage, :]] )
                b2line.set_data( [txb2[id_tige, nbimage, :], tyb2[id_tige, nbimage, :]] )
                #plot( [xi[0], xi[-1]], [yi[0], yi[-1]] , 'r--')
                draw()
            
            #print theta, xc, yc
            #Angle et projection pour le tir suivant ATTENTION AU MASQUE
            #0ld 1.4
            buffD = tdiams[id_tige,nbimage,:cpt]
            if len(buffD) > bufferangle:
                RR = .9 * buffD[-bufferangle:].mean()
            else:
                RR = .9 * buffD.mean()
            
            #Oldway
            x1n=xc - pas*sin(theta) - RR*cos(theta)
            y1n=yc - pas*cos(theta) + RR*sin(theta)
            x2n=xc - pas*sin(theta) + RR*cos(theta)
            y2n=yc - pas*cos(theta) - RR*sin(theta)
            

            dx = (x2n-x1n)/float(Np-1)
            dy = (y2n-y1n)/float(Np-1)
            
            xi = basexi*dx+x1n
            yi = basexi*dy+y1n

            #imp_local = local_contrast( imp, xc, yc )
            xc, yc, D, thetat, b1, b2, cgray = get_tige_border(xi, yi, imp, seuil_coupure=seuil_coupure)
            

            if xc and yc:
                
                #Save tige data
         
                add_tiges_pts(nbimage,id_tige, cpt, D, xc, yc, thetat, b1[0], b1[1], b2[0], b2[1], cgray )
                
                buffx = txcs[id_tige,nbimage,:cpt]
                buffy = tycs[id_tige,nbimage,:cpt]
                bufftheta = tthetas[id_tige,nbimage,:cpt]
                
                if len(buffx) > bufferangle:
                    #OLD VERSION RACINE SANS THETATMP just bufferanglemean ... car bug quand entre une certaine valeur 
                    #CAR singularité quand on passe de -180 a +180 (vers le bas aligné avec g !!!!) ou de +0 à -0
                    #BUG RESOLVED WITH CIRCMEAN 
                    
                    thetatmp = circmean( arctan2( -diff(buffx[-bufferangle/2:]), -diff(buffy[-bufferangle/2:]) ) )
                    theta = circmean( ma.hstack( [ bufftheta[-bufferangle:], thetatmp] ) )
                    #print theta
                    tthetas[id_tige,nbimage,cpt] = theta

                cpt +=1 
                
            else:
                passflag = False
    
            #coupure sur le rayon si trop petit
            if rayonfilter:
                buffR = tdiams[id_tige,nbimage,:cpt]
                if len(buffR) > 10:
                    Rmean = buffR[:-10].mean()
                else: 
                    Rmean = None
                    
                #Old 0.5 et 1.2
                if Rmean!=None and D!=None:
                    if D <= 0.5 * Rmean or D >= 1.2 * Rmean:
                        passflag = False
                        print("Interuption changement de rayon R=%0.2f moy=%0.2f"%(D,Rmean))
                
            if cpt >= Max_iter:
                passflag = False
                print("Iterations coupure")

            if not passflag:
                #Stop iteration
                break 


###############################################################################
    
def find_border_from_contour( polynome, xc, yc, slope, Np=100 ):
    
    #Table des points pour la recherche
    i = linspace(0.01,10,Np)
    outx = hstack( (xc - i, xc + i) )
    outy = (outx - xc ) * tan(slope) + yc
   
    iss = find( polynome.contains_points( zip(outx, outy) )  )

    if iss != []:
        outx = outx[iss]
        outy = outy[iss]
        
        nxc = outx.mean()
        nyc = outy.mean()
        
        b1 = (outx[0], outy[0])
        b2 = (outx[-1], outy[-1])
        
        diam = sqrt( (b2[0]-b1[0])**2 + (b2[1]-b1[1])**2 )
        
        #POUR VOIR EN DIRECT 
        #plot(outx,outy,'k-',lw=2)
        
    else:
        outx = []
        outy = []
        nxc, nyc, b1, b2, diam = None,None,None,None,None
        
    return nxc, nyc, diam, slope, b1, b2


def methode_Hugo( image, tiges_table, id_tige, nbimage, xi, yi, pas, Np, seuil_coupure=0.2, show_tige=False, rayonfilter=True ):
    from matplotlib.path import Path
    from skimage import measure
    
    #Max iteration
    Max_iter = 2000
    bufferangle = int(3/pas)
    passflag = True

    #Astuce pour petit grain de temps dans la boucle 
    add_tiges_pts = tiges_table.add_point 
    tdiams = tiges_table.diam
    txcs = tiges_table.xc
    tycs = tiges_table.yc
    tthetas = tiges_table.theta
    
    #On trouve le contour
    imgsl = [yi.mean()-400,yi.mean()+400,xi.mean()-400,xi.mean()+400]
    xoff = imgsl[2]
    yoff = imgsl[0]
    contours = measure.find_contours(image[imgsl[0]:imgsl[1],imgsl[2]:imgsl[3]], seuil_coupure)
    xi = xi-xoff
    yi = yi-yoff
    
    #On garde que le contour qui contient la base
    xcont, ycont = None, None
    ptsbase = zip(xi,yi)
    for contour in contours:
        if len(contour) > 100:
            xcont = contour[:, 1]
            ycont = contour[:, 0]
    
            p0 = Path( zip(xcont, ycont), readonly=True)
            ipts = find(p0.contains_points( ptsbase ))   
            
            if len(ipts) > 2 and len(ipts)<=len(xi)*0.7:

                xncont, yncont = cut_contour(xcont, ycont, xi[ipts], yi[ipts], epsi=5)
                if show_tige:    
                    plot(xncont,yncont)
                    
                #Interpolation du contour
                t = linspace(0, 1, len(xncont))
                t2 = linspace(0, 1, 500)

                x2 = np.interp(t2, t, xncont)
                y2 = np.interp(t2, t, yncont)

                
                p0 = Path( zip(x2, y2), readonly=True)
                xp = x2
                yp = y2
                break
            
            else:
                xcont = None
                ycont = None

                
    #Si pas de contour trouvé on stop
    if xcont == None and ycont == None:
        return False
    
    #Le premier points
    cpt = 0
    xi = xi[ipts]
    yi = yi[ipts]
    s = arctan2((yi[-1]-yi[0]),(xi[-1]-xi[0]))
    xnc, ync = xi.mean(), yi.mean()
    
    if show_tige:
        plot(xnc, ync, 'ro')   

    #Boucle until la fin de la tige
    for i in xrange(Max_iter-1):
        xc, yc, D, theta, b1, b2 = find_border_from_contour( p0, xnc, ync, s, Np)
        cgray = 0 #TODO

        if xc != None:
            
            #Projection du nouveau centre
            xnc = xc-cos(theta+pi/2.)*pas
            ync = yc-sin(theta+pi/2.)*pas

            #Calcul de la nouvelle pente (moyenne de la direction des centres)
            buffx = txcs[id_tige,nbimage,:cpt]
            buffy = tycs[id_tige,nbimage,:cpt]
            bufftheta = tthetas[id_tige,nbimage,:cpt]

            if len(buffx) > bufferangle:
                stmp = pi/2. + circmean( arctan2( diff(buffy[-bufferangle/2:]), diff(buffx[-bufferangle/2:]) ) )
                s = circmean( hstack( bufftheta[-bufferangle:], stmp ) )
                tthetas[id_tige,nbimage,cpt] = s
                #s = s
                #s = pi/2 + arctan2((ync[-1]-mean(ync[:-10])),(xnc[-1]-mean(xnc[:-10])))
                
        
            add_tiges_pts(nbimage,id_tige, cpt, D, xc+xoff, yc+yoff, theta, b1[0]+xoff, b1[1]+yoff, b2[0]+xoff, b2[1]+yoff, cgray ) 
            cpt += 1
            
        else:
            passflag = False
            
        if not passflag:
            break
    
###############################################################################
    
def traite_une_image( image, xypoints, imgnum, tiges, pas = 0.3, seuil="auto", Np = 100, show_tige = False, rois = None, gains = 20 , cut_off = 0.2, disk_size=4, rayonfilter=False, method="Olivier", image_class=None ):
    """
        Fonction pour traiter une image 
    """    
    
    t = time.time()
    image_class.load( image )
    imgF = image_class.render()
    #imgF = image_class.get_image(imgnum)
    imgnumF = 0
    print('load image in %f s'%(time.time()-t))
    
    #Si on a des zone d'interets pour faire un redressement de contraste local
    if rois != None:
        imgF = adjust_image(imgF, rois, gains, cut_off, disk_size)
            
    #Extraction des images
    for i in xrange( len(xypoints) ):
        xystart = xypoints[i]
        xi, yi= linspace(xystart[0][0], xystart[1][0], Np, dtype='float32'), linspace(xystart[0][1], xystart[1][1], Np, dtype='float32')
        if method == "Olivier":
            if seuil == "auto":
                seuiln = auto_seuil(imgF, xi, yi)
            else:
                seuiln = seuil
            methode_Olivier( imgF, tiges, i, imgnum, xi, yi, pas, Np, seuiln, show_tige, rayonfilter)
        
        if method == "Hugo":
            seuiln = auto_seuil_contour(imgF, xi, yi)
            methode_Hugo(imgF, tiges, i, imgnum, xi, yi, pas, Np, seuiln, show_tige, rayonfilter)
        #tige = extract_tige(  imgF, xypoints[i], id_tige=i, show_tige=show_tige, Np = Np, seuil_coupure = seuil, pas=pas, method=method, rayonfilter=rayonfilter  ) 
        #tiges.add( imgnum, tige )
        
    return {'imgname': image, 'imgnum': imgnumF, 'iimg': imgnum}      
    
def traite_une_image_thread( Queue_images, Queue_tiges, Tigesdata, xypoints, pas = 0.3, seuil="auto", Np = 100, show_tige = False, rois = None, gains = 20, cut_off = 0.2, disk_size=4, rayonfilter=False, method="Olivier", image=None ):
    """
        Fonction pour traiter une image 
    """    
    
    #Iteration tant que pas STOP dans la liste d'images
    imreadfunc = image.load
    imrendfunc = image.render
    
    try :
        for imgT in iter(Queue_images.get, 'STOP'):
            img = imgT[1]
            iimg = imgT[0]
            imreadfunc( img )
            imgF = imrendfunc()
            #imgF = image.get_image( iimg )
            #print(imgF)
            imgnumF = 0


            #Si on a des zone d'interets pour faire un redressement de contraste local
            if rois != None:
                imgF = adjust_image(imgF, rois, gains, cut_off, disk_size)
                    
                    
            #print("test")
            for i in xrange( len(xypoints) ):
                #tt[i] = extract_tige(  imgF, xypoints[i], id_tige=i, show_tige=show_tige, Np = Np, seuil_coupure = seuil, pas=pas, method=method, rayonfilter=rayonfilter )
                xystart = xypoints[i]
                xi, yi= linspace(xystart[0][0], xystart[1][0], Np, dtype='float32'), linspace(xystart[0][1], xystart[1][1], Np, dtype='float32')
                if method == "Olivier":
                    if seuil == "auto":
                        seuiln = auto_seuil(imgF, xi, yi)
                    else:
                        seuiln = seuil
                    methode_Olivier( imgF, Tigesdata, i, iimg, xi, yi, pas, Np, seuiln, show_tige, rayonfilter)
                
                if method == "Hugo":
                    seuiln = auto_seuil_contour(imgF, xi, yi)
                    methode_Hugo(imgF, Tigesdata, i, iimg, xi, yi, pas, Np, seuiln, show_tige, rayonfilter)
                
            Queue_tiges[iimg] = {'imgname': img, 'imgnum': imgnumF, 'iimg': iimg} 
            
    except Exception, e:
       print("Failed with: %s" % (e.message))
       print(iimg)
       
    return True 

def adjust_image(image, rois, gains, cut_off, disk_size):
    """
        Fonction pour aplliquer une correction de courbe (courbe sigmoid de skimage) sur certaines zones de l'image
        définies par les Region Of Interes (rois).
        Un filtre morph_open permet de diminuer les poussieres sur l'image
    """    
    imgF = image
    
    #Les test pour savoir si on definit un int ou une list (different seuil et gain pour les differents rois)
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

        imgF[roi[0]:roi[1],roi[2]:roi[3]] = adjust_sigmoid( imgF[roi[0]:roi[1],roi[2]:roi[3]], curcut_off, curgains )                
        #Netoyage
        if _iscv2:
            imgF[roi[0]:roi[1],roi[2]:roi[3]] = cv2.morphologyEx(  imgF[roi[0]:roi[1],roi[2]:roi[3]], cv2.MORPH_OPEN, disk(disk_size))
        
    return imgF

def traite_transitoire(file_names, num_images, num_tiges, pas = 0.3, seuil="auto",
                        Np = 100, thread=False, show_tige = False, base_points = None,
                        rois = None, gains = 20, cut_off = 0.2, disk_size=4, rayonfilter=False, 
                        method="Olivier", use_bw=True, color_transform=None, color_band=None ):
                            
    """
        For compatibility with old version
    """
    
    return Process_images( file_names, num_images, num_tiges, pas, seuil,
                        Np, thread, show_tige , base_points ,
                        rois , gains, cut_off, disk_size, rayonfilter, 
                        method, use_bw, color_transform, color_band )
                        

def methode_Olivier_test(Queue_tiges, imgt, iimg, ttable, pas, Np):
    #print(argss['seuil'],imgt.shape)
    try:
        for tt in iter(Queue_tiges.get, 'STOP'):
            itige, xi, yi, seuil = tt
            methode_Olivier(imgt, ttable, itige, iimg, xi, yi, pas, Np, seuil, rayonfilter=False)  
    except Exception, e:
       print("Failed with: %s" % (e.message))
       
        
    return True

def test_thread_on_tige(file_names, base_points, output_function, data_out):
    Np = 100
    pas = 0.3
    idtiges = arange(len(base_points))
    ttable = Tiges( len(base_points), len(file_names), size=2000, thread_safe=True)
    
    ta = time.time()
    
    xis = []
    yis = []
    for i in idtiges:
        #tt[i] = extract_tige(  imgF, xypoints[i], id_tige=i, show_tige=show_tige, Np = Np, seuil_coupure = seuil, pas=pas, method=method, rayonfilter=rayonfilter )
        xystart = base_points[i]
        xi, yi = linspace(xystart[0][0], xystart[1][0], Np, dtype='float32'), linspace(xystart[0][1], xystart[1][1], Np, dtype='float32')
        xis += [xi]
        yis += [yi]

    for iimg, name in enumerate(file_names):
        output_function(iimg)
        #load image
        imgt = cv2.imread(name, 0)/255.
        

        queue = mp.Queue() #Liste des tiges a traite
        [ queue.put( (i, xis[i], yis[i], auto_seuil(imgt, xis[i], yis[i])) ) for i in xrange(len(xis)) ]
        
        processes = []
        for w in xrange(4):
            p = mp.Process( target=methode_Olivier_test, args=(queue, imgt, iimg, ttable, pas, Np) )
            processes.append( p )
            p.start()
            queue.put('STOP') #Ajout du stop de fin BESOIN DE LE FAIRE A LA FIN DES PROCESS
        
        for ps in processes:
            ps.join()
        
        #print res
    
    
    ttable.Mask_invalid()
    print('Done in %f s'%(time.time()-ta))
    #print(ttable.xc)
    output = {'tiges_data':ttable, 'tiges_info': None}
    data_out.put( {"data": output, "imgs": file_names, "xypoints": base_points} )
    return 
    
def default_output_print(imnum):
    print("Traitement de %0.2f \%"%imnum) 
                   
def Process_images( file_names, num_images, num_tiges, pas = 0.3, seuil="auto",
                        Np = 100, thread=False, show_tige = False, base_points = None,
                        rois = None, gains = 20, cut_off = 0.2, disk_size=4, rayonfilter=False, 
                        method="Olivier", use_bw=True, color_transform=None, color_band=None, 
                        output_function=default_output_print, outputdata=None):
    """
    Fonction pour lancer le traitement des tiges par recherche de maximum
    
    Arguments
    ---------
    
    -file_names: nom des fichier images a charger (style unix on peut utiliser *,? etc) expl: './DSC_*.JPG'
                 or a list of files ['file1.jpg','file2.jpg',etc...]
    
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
    
    """
    
    #Creation d'objet Image qui contient les specifications de transformation de l'image a traiter
    image = Image(use_bw=use_bw, color_transform=color_transform, color_band=color_band)
    
    

    
    #Ouverture des images 
    #Check si c'est une liste ou une commande de recherche type unix 
    if type(file_names) != type(""):
        imgs = file_names    
    else:
        try:
            imgs = sorted( glob.glob( file_names ), key=lambda x: int(''.join(finddigits.findall(x.split('/')[-1]))) )
        except:
            imgs = sorted( glob.glob( file_names ) )
        
    #imgs = io.ImageCollection('/home/hugo/developpement/python/testgravitro/Hugo/test/manip11_07_14/apres2/*.JPG')
    if num_images == 'all':
        num_images = len( imgs )
      
    #Gestion des listes
    if type(num_images) == type([]) and len(num_images) == 2:
        img_start = num_images[0]
        img_stop = num_images[1]
        
        if img_stop == 'end':
            img_stop = len( imgs )

        
        imgs = imgs[img_start:img_stop]
        print("traitement de %s -> %s"%(imgs[0], imgs[-1]))
        num_images = len(imgs)
    
    #Creation de la db en hdf5
    #image = Images_hdf(imgs)
    #image.create_db()
      
    Num_tiges = num_tiges
    #print(Num_tiges, num_images)
    
    #Premierre image = selection base tige si pas de points de base des iges données
    if not base_points:
        #image.load( imgs[0] )
        #img = image.render()
        img = image.get_image(0)
        close('all')
        fig = figure('img')
         
        #Crop and stack
        #im = hstack( crop_image( img ) )
        imshow(img, cmap=cm.gray)
        
        #tight_layout()    
        xypoints = []
        ax = gca()
        ax.set_autoscale_on(False)
        for i in xrange(Num_tiges):
            #Trace un trait
            xypoints += [ ginput(n=0, timeout=0) ]
            plot( [ xypoints[-1][0][0], xypoints[-1][1][0] ], [ xypoints[-1][0][1], xypoints[-1][1][1] ], 'ro-', lw=2 )
            draw()
        
        if not show_tige:
            close('all')
    else:
        xypoints = base_points
    ##############

    #Lancement du traitement avec ou sans threads
    ta = time.time()
    results = []
    Tigesdata = Tiges( len(xypoints), num_images, size=2000, thread_safe=thread)
    #tiges = Tiges_hdf('./test.h5', len(xypoints), num_images)
    #tiges.create_db()
    
    if thread:
        
        worker = mp.cpu_count() #Nombre de processeurs 
        queue = mp.Manager().Queue() #Liste des images a traiter 
        outputs = mp.Manager().dict() #Liste des tiges pour les enregistrer
        [ queue.put( (inum, imgs[inum]) ) for inum in xrange( num_images )  ] #On peuple la la liste d'images
        processes = [] #Pour enregistrer les differents processus lancées en meme temps
        
        for w in xrange(worker):
            p = mp.Process( target=traite_une_image_thread, args=(queue, outputs, Tigesdata, xypoints, pas, seuil, Np, show_tige, rois, gains, cut_off, disk_size, rayonfilter, method, image ) )
            processes.append( p )
            p.start()
            queue.put('STOP') #Ajout du stop de fin BESOIN DE LE FAIRE A LA FIN DES PROCESS
            
        #Affichage de l'avencement que pour ipython
        if _isnotebook:
            pb = HTML(
            """
            <h3>Images traitées: <div class="progress-label" style="display:inline-block;"></div>%</h3>
            <div id="progressbar"></div> 
            """)
            display(pb)
        
        while any(i.is_alive() for i in processes) and queue.qsize() > 0:
            time.sleep(0.2)
            i = ( float(len(outputs))/num_images ) * 100
            if _isnotebook:
                display(Javascript('$( "#progressbar" ).progressbar({value: %0.2f}); $( ".progress-label" ).text( $( "#progressbar" ).progressbar( "value" ) )'%(i)))
            else:
                output_function(len(outputs))

        for ps in processes:
            ps.join()
        
        if _isnotebook:        
            #clean the cell
            clear_output()
            del pb
             
        results = [r[1] for r in outputs.items()]
        results.sort(key = lambda x: x['iimg'])
        Tigesdata.Mask_invalid() #Mask invalid data
            
       
        del outputs
        del queue

    else:
        #ICI C'est quand utilise pas le calcul parallel
        #Affichage de l'avencement que pour ipython
        if _isnotebook:
            pb = HTML(
            """
            <h3>Images traitées: <div class="progress-label" style="display:inline-block;"></div>%</h3>
            <div id="progressbar"></div> 
            """)
            display(pb)
        
        for imnum in xrange( num_images ):
            #print("traite image %i sur %i"%(imnum+1, num_images))
            i = ( imnum/float(num_images) ) * 100
            if _isnotebook:
                display(Javascript('$( "#progressbar" ).progressbar({value: %0.2f}); $( ".progress-label" ).text( $( "#progressbar" ).progressbar( "value" ) )'%(i)))
            else:
                output_function(imnum)
            results += [ traite_une_image( imgs[imnum], xypoints, imnum, Tigesdata, pas, seuil, Np,  show_tige, rois, gains, cut_off, disk_size, rayonfilter, method, image) ]
            
        if _isnotebook:
            #clean the cell
            clear_output()
        
    print("Done in %f s"%(time.time() - ta))
    
    
    output = {'tiges_data':Tigesdata, 'tiges_info': results}
    if outputdata != None:
        print('store data')
        outputdata.put( {"data": output, "imgs": imgs, "xypoints": xypoints} )
        return
    else:
        return output, imgs, xypoints

    
###############################################################################
#  Fonction diverses
###############################################################################
    
def quick_check( results, time_position ):
    """
        Verifier la detection des tiges sur une image a un temps donné
    """
    n = time_position
    
    imgname = results['tiges_info'][n]['imgname']
    
    imshow(Image.open(imgname).convert('L')/255., cmap=cm.gray )
    plot(results['tiges_data'].xc[:,n,:].T,results['tiges_data'].yc[:,n,:].T,lw=2)
    axis('equal')
    
def auto_seuil(image, xi, yi):
    """Fonction pour determiner automatiquement le seuil a partir du premier profil"""

    #ligne de gris
    if _iscv2:
        zi = cv2.remap(image, xi, yi, cv2.INTER_LINEAR)[:,0]
    else:
        zi = ndimage.map_coordinates(image.T, vstack( (xi, yi) ), order=1 )
    
    #print zi.max(), zi.min(), zi.max()-zi.min()
    #Condition si il y a une tige ou pas 
    if (zi.max()-zi.min()) < 0.05:
        #pas de tige ont met le seuil a 1 ce qui coupe la detection
        seuil = 1.0
    else:
        seuil = (zi.max()-zi.min()) * 0.3
    
    return seuil

def cut_contour( xcont, ycont, xbase, ybase, epsi=1):
    #Fonction pour couper le contour a la base
    
    #Pente de la normale
    s = arctan2((ybase[-1]-ybase[0]),(xbase[-1]-xbase[0])) - pi/2.
    xc, yc = xbase.mean(), ybase.mean()
    
    pts1 = (xc,yc)
    pts2 = (xc+(sign(s)*xc), (xc+(sign(s)*xc) - xc ) * tan(s) + yc)
    
    norm = sqrt( (pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2 )
    
    #plot( [pts1[0], pts2[0]], [pts1[1], pts2[1]], lw=3 )
    #On projete les points du contour par rapport a la droite normale a la base
    #http://paulbourke.net/geometry/pointlineplane/
    uproj = ( (xcont - pts1[0]) * (pts2[0]-pts1[0]) + (ycont - pts1[1]) * (pts2[1]-pts1[1]) ) / norm**2
    
    nxcont = pts1[0] + uproj * (pts2[0]-pts1[0])
    #TODO: CHECK SI LE SIGNE EST TOUJOURS BON QUAND TIGE VERS LE BAS-DROIT
    igood = find(nxcont <= pts1[0]+epsi)
    
    xcont = xcont[igood]
    ycont = ycont[igood]
    
    #Remove les points trop haut ou bas par rapport a la base
    #Projection le long de la ligne de base
    wproj = ( (xcont-xbase[0]) * (xbase[-1]-xbase[0]) + (ycont-ybase[0]) * (ybase[-1]-ybase[0]) ) / ( (xbase[-1]-xbase[0])**2 + (ybase[-1]-ybase[0])**2 )
    nycont = ybase[0] + wproj * (ybase[-1]-ybase[0])
    
    jgood = find( ( nycont >= ybase.min() - epsi ) & ( nycont <= ybase.max() + epsi ) )
    
    return xcont[jgood], ycont[jgood]

def auto_seuil_contour(image, xi, yi):
    """Fonction pour determiner automatiquement le seuil a partir du premier profil"""

    #ligne de gris    
    if _iscv2:
        zi = cv2.remap(image, xi.astype('float32'), yi.astype('float32'), cv2.INTER_LINEAR)[:,0]
    else:
        zi = ndimage.map_coordinates(image.T, vstack( (xi, yi) ), order=1 )
    
    seuil = zi.mean()
    
    return seuil
    
def return_cercle( rayon, xc, yc ):
    """
        Fonction pour retourner les coordonnees d'un cercle 
    """
    
    xi, yi = [], []
    for o in linspace(0,2*pi, 100):
        xi += [ rayon * cos(o) ]
        yi += [ rayon * sin(o) ]
        
    return array(xi) + xc, array(yi) + yc
    
def save_results ( results, nameout ):
 
    #Dump
    with open(nameout,"wb") as out:
        pickle.dump( results, out, protocol = 2 )

        
def load_results ( file_name ):
    
    with open( file_name, 'rb' ) as fin:
        data = pickle.load( fin )

    
    return data

def load_file( dir=None, multi= None ):
    """
    Select a file via a dialog and returns the file name.
    """
    try:
        from PySide import QtCore, QtGui
    except ImportError:
        from PyQt4 import QtCore, QtGui

    if dir is None: dir ='./'
    
    #filter="All files (*);; SM Files (*.sm)"
    fname = QtGui.QFileDialog.getOpenFileName(None, "Select data file...", dir)
    
    if multi:
        diro = '/'.join( str(fname).split('/')[:-1] ) + '/'
        ext = str(fname).split('.')[-1]
        out = diro + '*.' + ext
    else:
        out = str(fname)
        
    return out



class check_results():
    """
        Class pour afficher image par image les resultat
    """      
    
    def __init__(self, results ):
        
        self.imgs = [ i['imgname'] for i in results['tiges_info'] ]
        self.idimg = [ i['iimg'] for i in results['tiges_info'] ]
        self.tiges = results['tiges_data']
        
        #Initialise la figure
        self.cpt = 0
        self.imF = cv2.imread( self.imgs[self.cpt] )
        self.add_tiges()
        cv2.startWindowThread() 
        cv2.namedWindow('img')
        # create trackbars for color change
        cv2.createTrackbar('cpt','img',self.cpt,len(self.idimg)-1,self.updatef)
        r = 0.5
        dim = ( int(self.imF.shape[1] * r), int(self.imF.shape[0] * r))
        
        while True:	
            resized = cv2.resize( self.imF, dim)
            cv2.imshow('img', resized )
            self.cpt = cv2.getTrackbarPos('cpt','img')
            k = cv2.waitKey(1) & 0xFF 
            if k == ord('q'):     
                break
            elif k == 83: #fleche de droite
                self.nexte()
            elif k == 81: #fleche de gauche
                self.preve()
                s

    def updatef(self,x):
        self.imF = cv2.imread( self.imgs[self.cpt] )            
        self.add_tiges()
        
    def nexte(self):
        if self.cpt != len( self.idimg ) - 1 :
            self.cpt += 1
            self.updatef(None)
    
    def preve(self):
        if self.cpt != 0:
            self.cpt -= 1
            self.updatef(None)
            

    def add_tiges(self):
        for i in xrange( len(self.tiges.xc[:,0,0]) ):
            lines = array(vstack([self.tiges.xc[i,self.cpt,~self.tiges.xc[i,self.cpt,:].mask], self.tiges.yc[i,self.cpt,~self.tiges.xc[i,self.cpt,:].mask]]).T, int32)
            lines = lines.reshape(-1,1,2)
            cv2.polylines(self.imF,[lines],False,(0,255,0),3)

def local_contrast( image, xc, yc, size=30, gain=30, cutoff=0.15 ):
    """
    Fonction pour renforcer localement le contrast autour du centre actuel
    """
    
    #Calcul des indices sur l'image pour faire l'ajustement du contrast
    xcercle, ycercle = return_cercle( size, xc, yc )
    imin = int(min(xcercle)) 
    imax = int(max(xcercle)) 
    jmin = int(min(ycercle)) 
    jmax = int(max(ycercle)) 

    #Ajustement de l'image
    image[jmin:jmax,imin:imax] = adjust_sigmoid( image[jmin:jmax,imin:imax], cutoff, gain )

    return image
       
###############################################################################
#                     TRAITEMENT DES TIGES EXTRAITES                          #
###############################################################################
def moving_avg(x, W):    
    #La lenteur
    #S = array( [ mean( x[i-W/2:i+W/2] ) for i in arange(W/2, len(x)-W/2)] )    
    xT = ma.hstack( [ x[W/2-1:0:-1], x, x[-1:-W/2 - 1:-1] ] )
    wind = ones( W )    
    S = convolve(wind/wind.sum(), xT, mode='valid')
    return S
    
def traite_tige( xc, yc, R, pas ):
    """
        Fonction pour traiter une seul tige
        
        tige
        ----
        Dictionnaire contenant les infos d'une tige (cf fonction tige2dict pour convertir la class tige en dict)
    """
    
    xc = xc[~xc.mask] #prend que les donnees valides
    yc = yc[~yc.mask]
    R = R[~R.mask]
    
    
    #base de la tige a zero
    x = xc - xc[0] 
    y = yc - yc[0] 
    
    #Taille de la tige 
    Tt = sqrt( x[-1]**2 + y[-1]**2)
    
    #Filtre les tiges trop petites
    if Tt > 1.5 * R.mean():
        #Retourne y car le top de la photo est en haut
        y = -y
        
        #Rayon moyen pour estimer la taille du smooth
        N = round( R.mean()/pas )
        sx = moving_avg( x, N )
        sy = moving_avg( y, N )
        
        #Add zero at the begining (because moving avg remove wind/2 pts at stat and wind/2 pts at the end)
        sx = hstack((x[0], sx))
        sy = hstack((y[0], sy))
        
        dx, dy= diff(sx), diff(sy)
        ds = sqrt( dx**2 + dy**2 )
        s = cumsum( ds )
        
        T = rad2deg( arctan2( -dx, dy) )
    else:
        sx, sy, T, s, N = None, None, None, None, None
        
    return sx, sy, T, s, N
    
def traite_tige2( xc, yc, R, pas, coupe=5 ):
    """
        Fonction pour traiter une seul tige
        
        tige
        ----
        Dictionnaire contenant les infos d'une tige (cf fonction tige2dict pour convertir la class tige en dict)
        
        coupe
        -----
        enleve x points à la fin de la detection de la tiger 
        
    """
    
    xc = xc[~xc.mask] #prend que les donnees valides
    yc = yc[~yc.mask]
    R = R[~R.mask]
    
    
    
    #base de la tige a zero
    x = xc - xc[0] 
    y = yc - yc[0]

    #Taille de la tige 
    Tt = sqrt( x[-1]**2 + y[-1]**2)
    
    #Filtre les tiges trop petites
    if Tt > 1.5 * R.mean() and len(x)>coupe*2:
        #Retire coupe point au bout
        x = x[:-coupe]
        y = y[:-coupe]
        R = R[:-coupe]
    
        #Retourne y car photo axes dans autre sens
        y = -y
        
        #Rayon moyen pour estimer la taille du smooth
        N = round( R.mean()/pas )
        sx = moving_avg( x, N )
        sy = moving_avg( y, N )
        
        #Add zero at the begining (because moving avg remove wind/2 pts at stat and wind/2 pts at the end)
        #sx = hstack((x[0], sx))
        #sy = hstack((y[0], sy))
        
        
        dx, dy= diff(sx), diff(sy)
        
        ds = sqrt( dx**2 + dy**2 )
        s = cumsum( ds )
        

        T = arctan2( -dx, dy) 
        
      
        
    else:
        sx, sy, T, s, N = None, None, None, None, None
        
    return sx, sy, T, s, N
 
def traite_tiges( tiges, itige = None, pas = 0.3, causette=True):
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
        
    """
    
    if not itige:
        Ntige = len( tiges.xc ) 
        itige = range( Ntige )
    else:
        Ntige = len( itige )


    lines = [ None ] * Ntige

    
    bad_value = -30000
    Ntime = len(tiges.xc[0,:,0])
    Max_tige_lengh = 3000 #Taille de la convolution possible
    allx = ma.masked_equal( zeros( (Ntige, Ntime, Max_tige_lengh) ) + bad_value, bad_value )
    ally = ma.masked_equal( zeros_like(allx) + bad_value, bad_value )
    L = ma.masked_equal( zeros((Ntige, Ntime)) + bad_value, bad_value )
    thetab = ma.masked_equal( zeros_like(L) + bad_value, bad_value )
    
    
    #boucle sur les tiges
    for i in xrange( Ntige ):
        linest = []
        
        #boucle sur le temps
        for t in xrange( Ntime ):      
            #Test si pas de detection des le debut
            tmp = ma.masked_invalid( tiges.xc[itige[i],t,:] )            
            if len(tmp[~tmp.mask]) > 1:
                xt, yt, theta, s, N = traite_tige( tiges.xc[itige[i],t,:], tiges.yc[itige[i],t,:], tiges.diam[itige[i],t,:]/2., pas )
                if xt != None and yt != None:      
                    di = round( 1.5 * N )
                    linest +=  [zip(xt, yt)]

                    thetab[ i, t ] = ma.mean(theta[-(N+di):-N])
                    L[ i, t ] = s[-1]

                        
                    #lines[ i, t, :len(xt) ] = zip(xt, yt)
                    allx[ i, t, :len(xt) ] = xt
                    ally[ i, t, :len(yt) ] = yt
                else:
                    if causette:
                        print("La taille de la tige %i sur l'image %i est trop faible"%(itige[i],t))
            else:
                if causette:
                    print( "La tige %i sur image %i a merdé"%(itige[i],t) )
            
        lines[i] = linest
        
        
    return allx, ally, L, thetab, lines
    
def traite_tiges2( tiges, itige = None, pas = 0.3, causette=True, return_estimation_zone=False):
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
        Ntige = len( tiges.xc ) 
        itige = range( Ntige )
    else:
        Ntige = len( itige )


    lines = [ None ] * Ntige

    
    bad_value = -30000
    Ntime = len(tiges.xc[0,:,0])
    Max_tige_lengh = len(tiges.xc[0,0,:]) #Taille de la convolution possible
    allx = ma.masked_equal( zeros( (Ntige, Ntime, Max_tige_lengh) ) + bad_value, bad_value )
    ally = ma.masked_equal( zeros_like(allx) + bad_value, bad_value )
    L = ma.masked_equal( zeros((Ntige, Ntime)) + bad_value, bad_value )
    thetab = ma.masked_equal( zeros_like(L) + bad_value, bad_value )
    measure_zone = [] #List to store indices where we do the mean of values
    
    #boucle sur les tiges
    for i in xrange( Ntige ):
        linest = []
        measure_zonet = []
        #boucle sur le temps
        for t in xrange( Ntime ):      
            #Test si pas de detection des le debut
            tmp = ma.masked_invalid( tiges.xc[itige[i],t,:] )            
            if len(tmp[~tmp.mask]) > 1:
                xt, yt, theta, s, N = traite_tige2( tiges.xc[itige[i],t,:], tiges.yc[itige[i],t,:], tiges.diam[itige[i],t,:]/2., pas )
                if (xt != None) and (yt != None):      
                    di = round( 1.5 * N )
                    measure_zonet += [ ( -(N+di), -N ) ]
                    x0,y0 = tiges.xc[itige[i],t,~tiges.xc[itige[i],t].mask][0], tiges.yc[itige[i],t,~tiges.yc[itige[i],t].mask][0]
                    linest +=  [zip(xt,yt)]

                    #thetab[ i, t ] = ma.mean(theta[-(N+di):-N])
                    #Reference 0 quand tige est verticale
                    thetab[ i, t ] = rad2deg( circmean(theta[-(N+di):-N], high=pi, low=-pi) ) 
                    
                    L[ i, t ] = s[-1]
                    #lines[ i, t, :len(xt) ] = zip(xt, yt)
                    allx[ i, t, :len(xt) ] = xt
                    ally[ i, t, :len(yt) ] = yt
                else:
                    if causette:
                        print("La taille de la tige %i sur l'image %i est trop faible"%(itige[i],t))
            else:
                if causette:
                    print( "La tige %i sur image %i a merdé"%(itige[i],t) )
            
        lines[i] = linest
        measure_zone += [measure_zonet]
        
    if return_estimation_zone:    
        return allx, ally, L, thetab, lines, measure_zone
    else:
        return allx, ally, L, thetab, lines

    
def plot_tiges( xt, yt, Lt, Tt, linest ):
    
    from mpl_toolkits.axes_grid1 import AxesGrid
    from matplotlib.collections import LineCollection

    fig = figure('tiges')
    
    Ntige = len(xt)
    G = GridSpec(3,Ntige)
    grid2 = AxesGrid(fig, G[0,:], # similar to subplot(142)
                    nrows_ncols = (1, Ntige),
                    axes_pad = 0.0,
                    share_all=True,
                    label_mode = "1",
                    cbar_location = "top",
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

        lcollec = LineCollection( linest[it], cmap=cm.PiYG_r, linewidth=(2,), alpha=1, norm=Normalize(vmin=cmin, vmax=cmax))
    
        lcollec.set_array( Tt[it] )
        
        grid2[it].add_collection( lcollec )
    
        
        
        
    maxX, minX = xt.max(), xt.min()
    maxY, minY = yt.max(), yt.min()
    
    
    grid2.cbar_axes[0].colorbar( lcollec )
    grid2.axes_llc.set_xlim( [minX, maxX] )
    grid2.axes_llc.set_ylim( [minY, maxY])
    grid2.axes_llc.set_xticks( [int(minX), 0, int(maxX)] )
    
    #Les moyennes
    axa1 = subplot( G[1,:] )
    plot( Lt.T )
    plot( Lt.mean(0), 'k', lw=3 )
    legend( [str(i) for i in xrange(Ntige)], loc=0, ncol=3, title='tiges', framealpha=0.3)
    subplot( G[2,:], sharex = axa1)
    plot( Tt.T )
    plot( Tt.mean(0), 'k', lw=3 )
    

def get_photo_time( image_path ):
    stat = pilimage.open(image_path)._getexif()
    ymd = stat[306].split(' ')[0].split(':')
    hms = stat[306].split(' ')[1].split(':')
    t = datetime.datetime(int(ymd[0]),int(ymd[1]),int(ymd[2]),int(hms[0]),int(hms[1]),int(hms[2]))
    #print str(stat[306])
    return t
    

def extract_Angle_Length( data_path, output_file_name, global_tige_id=None, image_path_mod=lambda pname: pname['imgname'], methode_de_traitements=traite_tiges2, xy=False, get_time_from_photos=True):
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
    
    #Manage type and convert inputs to list
    if type(data_path) != type([]):
        data_path = [data_path]
    if type(image_path_mod) != type([]):
        image_path_mod = [image_path_mod]
        
        if len(image_path_mod) != len(data_path):
            image_path_mod = [ image_path_mod[0] ] * len(data_path)
         


    #Global data
    if xy:
        data_out = pd.DataFrame(columns=['tige','angle','temps','taille','rayon','x','y','nom photo','sequence','angle_0_360'])   
    else:
        data_out = pd.DataFrame(columns=['tige','angle','temps','taille','rayon','sequence','angle_0_360'])   
    
    #loop over list (sequence)
    for i in range(len(data_path)):
        #Here is to get ride of windows path accent bugs
        try:
            print(u"Proceed %s"%data_path[i].decode('utf8'))
        except:
            print("Proceed data")
        #Load data
        data_in = load_results( data_path[i] )
        tiges = data_in['tiges_data']
        R = tiges.diam/2.0
        xt, yt, ll, aa, _ = methode_de_traitements( tiges )
        #Le temps a partir des photos    
        if get_time_from_photos:
            try:
                t = [get_photo_time( image_path_mod[i]( a ) ) for a in data_in['tiges_info']]
            except:
                print(u"No exif info on picturs to get time")
                t = arange(0,len(data_in['tiges_info']))
        else:
            t = arange(0,len(data_in['tiges_info']))
        
        #Loop over tige
        Ntiges = len(aa)
        for n in range(Ntiges):
            #Number of time records
            nstep = len(aa[n])
            
            #map tige id
            if global_tige_id != None:
                if n in global_tige_id[i]:
                    tigetmp = [global_tige_id[i][n]] * nstep
                else:
                    tigetmp = [n+1]*nstep
            else:
                tigetmp = [n+1]*nstep
                
            if xy:
                data_tmp = pd.DataFrame( {'tige': tigetmp, 'angle':aa[n], 
                                      'temps': t, 'taille': ll[n],
                                      'rayon': R[n].mean(1),
                                      'x': xt[n].tolist(),
                                      'y': yt[n].tolist(),
                                      'sequence': [i]*nstep,
                                      'angle_0_360': convert_angle(aa[n]),
                                      'nom photo': [os.path.basename(pnames['imgname']) for pnames in data_in['tiges_info']]})            
            else:
                data_tmp = pd.DataFrame( {'tige': tigetmp, 'angle':aa[n], 
                                      'temps': t, 'taille': ll[n],
                                      'rayon': R[n].mean(1),
                                      'sequence': [i]*nstep,
                                      'angle_0_360': convert_angle(aa[n])})
                                      
            data_out = data_out.append(data_tmp, ignore_index=True)
        
        
    print(data_out.head())
    
    #Save data to a csv
    if output_file_name != None:
        print(u"Saved to %s"%output_file_name)
        data_out.to_csv(output_file_name, index=False)
    else:
        return data_out


def convert_angle( angle ):
    #Convert angle from 0->180 and -0 -> -180 to 0->360
    return angle%360
    
def join_sequence(data, join_sec):
    """
        Function to make the join between section 
        
        join_sec -> dict[prev_sequence_id] = next_sequence_id
                    expl join sequence 1 et 2
                    {1:2}
    """

    import pandas as pd    
    
    data_out = data.copy()
    
    Nsec = data_out.sequence.unique()
    Ntige = data_out.tige.unique()
    
    for sec in Nsec:
        if sec in join_sec:
            for tige in data_out[ data_out.sequence == sec ].tige.unique():
                #Get next sequence first angle
                Astart = array(data_out.query('sequence == %i and tige == %i'%(join_sec[sec],tige)).angle)
                istart = find( pd.notnull(Astart) )[0]
                #Draw the sequence 
                angletmp = array(data_out.query('sequence == %i and tige == %i'%(sec,tige)).angle)
                iend = find( pd.notnull(angletmp) )[-1]
                
                dA = Astart[istart] - angletmp[iend]
                #print dA, Astart[istart]                
                if dA != NaN:
                    angletmp = angletmp + dA

                indexes = data_out.query('sequence == %i and tige == %i'%(sec,tige)).index
                data_out.angle[indexes] = angletmp
                
    return data_out


def plot_sequence(data, tige_color={1:"blue",2:"green",3:"red",4:"cyan",5:"orange",6:"magenta",7:"gray",8:"black",9:"sandybrown"}, show_lims=True, ydata='angle', tige_alpha=0.5):
    """
        Function to quickly plot sequence from pandas table
        
        tige_color -> dict[tige_id] = "color"
    """

    Nsec = data.sequence.unique()
    Ntige = data.tige.unique()

    #Automatic color    
    if tige_color == None:
        tige_color = cm.Set1( linspace(0, 1, len(Ntige)+1) )

    
    for i, tige in enumerate(Ntige):
        for sec in Nsec:
            #Try except because number of tige may change for each sequence
            try:
                data.query('sequence == %i and tige == %i'%(sec,tige)).plot('temps',ydata, ax=gca(), color=tige_color[i],alpha=tige_alpha, legend=False)
            except:
                pass
            
    #Plot the mean over
    for sec in Nsec:
        y_mean = data[ data.sequence == sec ].groupby('temps').mean()[ydata]
        plot(data[ data.sequence == sec ].temps.unique(), y_mean, color='RoyalBlue', lw=2)
        
        if show_lims:
            #Plot sec limites 
            yminmax = ylim()
            tmin = data[data.sequence == sec].temps.min()
            tmax = data[data.sequence == sec].temps.max()

            plot( [tmin, tmin], yminmax, '--', color='gray' )
            plot( [tmax, tmax], yminmax, '--', color='gray' )
            
    #ylabel
    ylabel(ydata)
       
###############################################################################

#Calcul de la longueur de croissance
def get_growth_length(tiges, cur_tige, thresold='auto', imgs = None, pas = 0.3):
    """
    Compute the curvature length as described in the AC model of Bastien et al.

    tiges: is the tiges instance   
    thresold[auto]: the thresold if computed as 2 times the mean diameter.
    imgs[None]: can be a list of two images to plot the initial and final position of the organ
    """
        
        
    #Need init time tige data
    ti_x, ti_y, ti_angle, ti_s, ti_N = traite_tige2( tiges.xc[cur_tige,0], tiges.yc[cur_tige,0], tiges.diam[cur_tige,0]/2.0, pas=pas)
    ti_xc, ti_yc = tiges.xc[cur_tige,0], tiges.yc[cur_tige,0] #raw tige xc,yc
    #print(diff(ti_s)[::20])
    
    #Need last time tige data
    tf_x, tf_y, tf_angle, tf_s, tf_N = traite_tige2( tiges.xc[cur_tige,-1], tiges.yc[cur_tige,-1], tiges.diam[cur_tige,-1]/2.0, pas=pas)
    tf_xc, tf_yc = tiges.xc[cur_tige,-1], tiges.yc[cur_tige,-1]
    #print(diff(tf_s)[::20])
    
    xmax = max((ti_xc.max(), tf_xc.max())) 
    xmin = min((ti_xc.min(), tf_xc.min())) 
    ymax = max((ti_yc.max(), tf_yc.max()))
    ymin = min((ti_yc.min(), tf_yc.min()))
    
    imgxma = int(xmax+0.02*xmax)
    imgxmi = int(xmin-0.02*xmin)
    imgyma = int(ymax+0.02*ymax)
    imgymi = int(ymin-0.02*ymin)
    
    #print((imgxmi, imgymi, imgxma-imgxmi, imgyma-imgymi))
    figure('Growth lenght')
    G = GridSpec(3,4)
    subplot(G[:2,2:])
    if imgs != None:
        imgi = pilimage.open(imgs[0]).crop( (imgxmi, imgymi, imgxma, imgyma) )
        imgf = pilimage.open(imgs[-1]).crop( (imgxmi, imgymi, imgxma, imgyma) )
        
        imshow(imgi.convert('L'), 'gray')
        imshow(imgf.convert('L'), 'gray', alpha=0.5)
        
    plot(tiges.xc[cur_tige,0]-imgxmi, tiges.yc[cur_tige,0]-imgymi, 'g-', lw=2)
    plot(tiges.xc[cur_tige,-1]-imgxmi, tiges.yc[cur_tige,-1]-imgymi, 'm--', lw=2)
    axis('equal')
    axis('off')
    #subplot(G[:2,0])
    #plot(ti_s, rad2deg(ti_angle))
    #plot(tf_s, rad2deg(tf_angle))
    
    #Caclul du spatio-temporel
    xt, yt = tiges.xc[cur_tige], tiges.yc[cur_tige]
    lx, ly = xt.shape
    dx, dy = diff(xt,1), diff(-yt,1)
    
    sdx, sdy = zeros_like(dx) - 3000, zeros_like(dy) - 3000
    W = round( (tiges.diam[cur_tige].mean()/2.)/pas ) 
    
    wind = ones( W, 'd' )    
    for i in xrange(dx.shape[0]):      
        dxT = ma.hstack( [ dx[i, W-1:0:-1], dx[i,:], dx[i,-1:-W:-1] ] )
        dyT = ma.hstack( [ dy[i, W-1:0:-1], dy[i,:], dy[i,-1:-W:-1] ] )
        cx=convolve(wind/wind.sum(), dxT, mode='valid')[(W/2-1):-(W/2)-1]
        cy=convolve(wind/wind.sum(), dyT, mode='valid')[(W/2-1):-(W/2)-1] 
        sdx[i,:len(cx)] = cx 
        sdy[i,:len(cy)] = cy
     
   
    sdx = ma.masked_less_equal(sdx, -100.0)
    sdy = ma.masked_less_equal(sdy, -100.0)
    Arad = ma.arctan2( -sdx, sdy )
    A = rad2deg( Arad )
    
    ax1 = subplot(G[0,:2])
    plot(A.mean(0), 'm', label='Average')
    plot(A[0,:], 'g', label='First')
    seuil = A[0,:len(A[0][~A[0].mask])/2].mean()
    xlims = ax1.get_xlim()
    plot(xlims,[seuil]*2, 'r--')
    
    #Find intersection with an offset
    try:
        ds_start = find( abs(A.mean(0)-seuil) > 5.0 )[0]
    except:
        ds_start = 0
        
    plot([ds_start]*2, ax1.get_ylim(), 'k--')
    
    legend(loc=0)
    
    ax2 = subplot(G[1,:2], sharex=ax1)
    pcolormesh(A)
    colorbar(orientation='horizontal', use_gridspec=True)
    plot([ds_start]*2, ax2.get_ylim(), 'k--')
    
    #Plot the growth zone (i.e. the one that curve)
    sinit = cumsum( sqrt( sdx[0]**2 + sdy[0]**2 ) )
    Si = sinit-sinit[0]
    Sc = sinit[ds_start:]-sinit[ds_start]
    AA0 = Arad[-1,ds_start:]/Arad[-1,ds_start]
    
        
    #Fit sur le log(A/A0)
    min_func = lambda p, x, y: sum( sqrt( (x*p[0] - y)**2 ) )
    opt_params = fmin(min_func, [ Sc[:len(Sc[~Sc.mask])].std()/AA0[:len(Sc[~Sc.mask])].std() ], 
                                 args = (Sc[~Sc.mask], log(AA0[~Sc.mask][:len(Sc[~Sc.mask])])))
    #fitA = poly1d( ma.polyfit(Sc, log(AA0), 1) )    
    
    #print(fitA)
    Lc = -1/opt_params[0]
    Lgz = max(Sc[~Sc.mask])
    Lo = max(Si[~Si.mask])
    GoodL = min((Lgz,Lo))
    B = GoodL/Lc
    print("Lc=%0.2f, Lzc=%0.2f, L0=%0.2f, B=%0.2f"%(Lc, Lgz, Lo, B))
    
    subplot(G[2,:2])
    plot(Sc[:len(Sc[~Sc.mask])], AA0[:len(Sc[~Sc.mask])], 'o')
    xtmp = linspace(0,max(gca().get_xlim()))
    plot(xtmp, exp(-xtmp/Lc), 'g', lw=2)
    
    subplot(G[2,2:])
    plot(Sc[:len(Sc[~Sc.mask])], log(AA0[:len(Sc[~Sc.mask])]), 'o')
    xtmp = linspace(0,max(gca().get_xlim()))
    plot(xtmp, -xtmp/Lc, 'g', lw=2)

    tight_layout()
    