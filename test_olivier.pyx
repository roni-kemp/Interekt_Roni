#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:15:52 2017

@author: hugo

Test pour convertir l'algorithm d'Olivier en cython pour l'accélerer !

"""


import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound 
from scipy import ndimage
from cmath import rect, phase
import cv2
_iscv2 = True

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.uint8_t DTYPE_tint
ctypedef np.float64_t DTYPE_tf64

@boundscheck(False)
@wraparound(False)
cdef float fast_circmean(np.ndarray rad):
    cdef int size = len(rad)
    cdef complex sum_rad = 0
    cdef float d
    
    for d in rad:
        sum_rad += rect(1, d)
        
    return phase(sum_rad/size)

@boundscheck(False)
@wraparound(False)
cdef (float, float, float, float, float, float, float, float, float) fast_get_tige_border(np.ndarray[DTYPE_t, ndim=1] xi,
     np.ndarray[DTYPE_t, ndim=1] yi, np.ndarray[DTYPE_tf64, ndim=2] image, float seuil_coupure=0.1):
    """
        Obtenir les bords de la tige a partir du profil

    """

    cdef float xcenterf, ycenterf, diam, theta, xb1, xb2, yb1, yb2
    cdef float bad_value = 30000.0
    cdef np.ndarray[DTYPE_tf64, ndim=1] zi = np.zeros_like(xi, dtype=np.float64)
    
    #zi = ndimage.map_coordinates(image.T, vstack( (xi, yi) ), order=1 )
    #cv2.INTER_LINEAR
    if _iscv2:
        zi = cv2.remap(image, xi, yi, cv2.INTER_LINEAR)[:,0]
    else:
        zi = ndimage.map_coordinates(image.T, np.vstack( (xi, yi) ), order=1, output=np.float64 )

    #figure('line')
    #plot( zi )
    ib1, ib2 = fast_get_min_max( zi, coupure=seuil_coupure )
    if ib1 == bad_value or ib2 == bad_value:
        xcenterf = bad_value
        ycenterf = bad_value
        diam = bad_value
        theta = bad_value
        xb1 = bad_value
        xb2 = bad_value
        yb1 = bad_value
        yb2 = bad_value
        cgray = bad_value
    else:

        #Calcul du centre et du rayon
        xb1, yb1 = xi[ib1], yi[ib1]
        xb2, yb2 = xi[ib2], yi[ib2]

        xcenterf, ycenterf = 0.5 * ( xb1 + xb2 ), 0.5 * ( yb1 + yb2 )
        diam = ( (xb2 - xb1) **2 + (yb1 - yb2)**2 )**0.5

        cgray = bad_value


        #L'angle de la pente
        theta = np.arctan2( -(yb2-yb1), xb2-xb1 )
        #theta = arctan2( -yb2+yb1, -(xb2-xb1) )



    return xcenterf, ycenterf, diam, theta, xb1, yb1, xb2, yb2, cgray

@boundscheck(False)
@wraparound(False)
cdef (int, int) fast_get_min_max(np.ndarray[DTYPE_tf64, ndim=1] z, float coupure=0.1):
    """
        Pour obtenir la position
        du maximum et du minimum de la derivee
        du profil de la tige z

    """

    cdef int ib1, ib2
    cdef int bad_value = 30000
    #On vire la moyenne entre min et max
    cdef float minz = z.min()
    cdef float maxz = z.max()
    cdef np.ndarray[DTYPE_tf64, ndim=1] zic =  z - 0.5 * ( minz + maxz )
    #cdef np.ndarray[DTYPE_t, ndim=1] fib1, fib2
    
    if ( maxz - minz ) < coupure:
        #print("cut ")
        ib1 = bad_value
        ib2 = bad_value
    else:
        #On interpole
        #zici = interp1d( arange( len(zic) ), zic )
        #Recup le gradient position du gradient max pour obtenir le bord
        gradz = np.gradient( np.sign(zic) )
        #On va chercher tous les pics positifs et negatifs
        fib1 = np.where( gradz == 1.0 )[0]
        fib2 = np.where( gradz == -1.0 )[0]
        
        if len(fib1) > 0 and len(fib2) > 0:
            ib1 = fib1[0]
            ib2 = fib2[len(fib2)-1]
        else:
            ib1 = bad_value
            ib2 = bad_value

    return ib1, ib2

@boundscheck(False)
cdef void compute_Olivier(np.ndarray[DTYPE_tf64, ndim=2] image, tiges_table, int id_tige, int nbimage,
                         np.ndarray[DTYPE_t, ndim=1] xi, np.ndarray[DTYPE_t, ndim=1] yi,
                         float pas, int Np, float seuil_coupure=0.2, int show_tige = False,
                         int rayonfilter=True):
    """
        Methode d'Olivier
    """
    
    
    cdef np.ndarray[DTYPE_t, ndim=1] basexi = np.arange(100, dtype=DTYPE)
    #Variables
    cdef int Max_iter = tiges_table.size
    #tige = Tige( id_tige, pas, size=Max_iter ) #Pour enregistrer les infos d'une tige
    cdef int cpt = 0
    cdef int bufferangle = int(3/pas)

    cdef DTYPE_tf64 bad_value = 30000.0
    cdef int passflag = True
    cdef np.ndarray[DTYPE_tf64, ndim=2] imp = image
    #Mon reglage avnt 0.9 et oliv 1.4
    cdef DTYPE_tf64 percent_diam = 0.9
    
    cdef np.ndarray[DTYPE_tf64, ndim=1] tdiams = np.empty([Max_iter], dtype=np.float64)
    cdef np.ndarray[DTYPE_tf64, ndim=1] txcs  = np.empty([Max_iter], dtype=np.float64)
    cdef np.ndarray[DTYPE_tf64, ndim=1] tycs  = np.empty([Max_iter], dtype=np.float64)
    cdef np.ndarray[DTYPE_tf64, ndim=1] tthetas  = np.empty([Max_iter], dtype=np.float64)
    
    cdef np.ndarray[DTYPE_tf64, ndim=1] txb1  = np.empty([Max_iter], dtype=np.float64)
    cdef np.ndarray[DTYPE_tf64, ndim=1] txb2  = np.empty([Max_iter], dtype=np.float64)
    cdef np.ndarray[DTYPE_tf64, ndim=1] tyb1  = np.empty([Max_iter], dtype=np.float64)
    cdef np.ndarray[DTYPE_tf64, ndim=1] tyb2  = np.empty([Max_iter], dtype=np.float64)
    
    cdef np.ndarray[DTYPE_tf64, ndim=1] buffx, buffy, bufftheta, buffR, buffD
    cdef DTYPE_tf64 xc ,yc, D, x1n, y1n, x2n, y2n, dx, dy, xb1, xb2, yb1, yb2, tmpsin, tmpcos
    cdef float theta, thetatmp, cgray
    cdef float size_xi = float(Np-1)
    #ny, nx = shape(image)
    #fi = RectBivariateSpline(arange(nx), arange(ny), image.T, kx=1, ky=1 )
    #imp_local = local_contrast( imp, mean(xi), mean(yi) )
    #Premier transect
    #print(xi,yi,imp,seuil_coupure)
    xc, yc, D, theta, xb1, yb1, xb2, yb2, cgray = fast_get_tige_border(xi, yi, imp, seuil_coupure=seuil_coupure)

    #plot(xni, yni,'r')
    #tige.add_point(cpt, D, xc, yc, theta, b1, b2, cgray)
    if xb1 != bad_value and xb2 != bad_value :
        #add_tiges_pts(nbimage,id_tige, cpt, D, xc, yc, theta, b1[0], b1[1], b2[0], b2[1], cgray )
        tdiams[cpt] = D
        txcs[cpt] = xc
        tycs[cpt] = yc
        tthetas[cpt] = theta
        txb1[cpt] = xb1
        tyb1[cpt] = yb1
        txb2[cpt] = xb2
        tyb2[cpt] = yb2
        
        cpt += 1

    #Boucle jusqu'au sommet
    if xc != bad_value and yc != bad_value:
        for i in range(Max_iter-1):

            #print theta, xc, yc
            #Angle et projection pour le tir suivant ATTENTION AU MASQUE
            #0ld 1.4
            #buffD = tdiams[id_tige,nbimage,:cpt]
            buffD = tdiams[:cpt]
            if len(buffD) > bufferangle:
                RR = percent_diam * buffD[-bufferangle:].mean()
            else:
                RR = percent_diam * buffD.mean()

            #Oldway
            tmpsin = np.sin(theta)
            tmpcos = np.cos(theta)
            
            x1n=xc - pas*tmpsin - RR*tmpcos
            y1n=yc - pas*tmpcos + RR*tmpsin
            x2n=xc - pas*tmpsin + RR*tmpcos
            y2n=yc - pas*tmpcos - RR*tmpsin


            dx = (x2n-x1n)/size_xi
            dy = (y2n-y1n)/size_xi

            xi = basexi*dx+x1n
            yi = basexi*dy+y1n

            #imp_local = local_contrast( imp, xc, yc )
            xc, yc, D, thetat, xb1, yb1, xb2, yb2, cgray = fast_get_tige_border(xi, yi, imp, seuil_coupure=seuil_coupure)


            if xc != bad_value and yc != bad_value:

                #Save tige data

                #add_tiges_pts(nbimage,id_tige, cpt, D, xc, yc, thetat, b1[0], b1[1], b2[0], b2[1], cgray )
                tdiams[cpt] = D
                txcs[cpt] = xc
                tycs[cpt] = yc
                tthetas[cpt] = theta
                txb1[cpt] = xb1
                tyb1[cpt] = yb1
                txb2[cpt] = xb2
                tyb2[cpt] = yb2
                

                #buffx = txcs[id_tige,nbimage,:cpt]
                #buffy = tycs[id_tige,nbimage,:cpt]
                #bufftheta = tthetas[id_tige,nbimage,:cpt]
                buffx = txcs[:cpt]
                buffy = tycs[:cpt]
                bufftheta = tthetas[:cpt]
                
                if len(buffx) > bufferangle:
                    #OLD VERSION RACINE SANS THETATMP just bufferanglemean ... car bug quand entre une certaine valeur
                    #CAR singularité quand on passe de -180 a +180 (vers le bas aligné avec g !!!!) ou de +0 à -0
                    #BUG RESOLVED WITH CIRCMEAN

                    thetatmp = fast_circmean( np.arctan2( -np.diff(buffx[-bufferangle/2:]), -np.diff(buffy[-bufferangle/2:]) ) )
                    theta = fast_circmean( np.ma.hstack( [ bufftheta[-bufferangle:], thetatmp] ) )
                    #print theta
                    #tthetas[id_tige,nbimage,cpt] = theta
                    tthetas[cpt] = theta

                cpt +=1

            else:
                passflag = False

            #coupure sur le rayon si trop petit
            if rayonfilter:
                #buffR = tdiams[id_tige,nbimage,:cpt]
                buffR = tdiams[:cpt]
                if len(buffR) > 10:
                    Rmean = buffR[:-10].mean()
                else:
                    Rmean = None

                #Old 0.5 et 1.2
                if Rmean!=None and D!=bad_value:
                    if D <= 0.5 * Rmean or D >= 1.2 * Rmean:
                        passflag = False
                        print("Interuption changement de rayon R=%0.2f moy=%0.2f"%(D,Rmean))

            if cpt >= Max_iter:
                passflag = False
                print("Iterations coupure")


            if not passflag:
                #Stop iteration
                break
            
    
        tiges_table.diam[id_tige, nbimage,:cpt] = tdiams[:cpt]
        tiges_table.xc[id_tige, nbimage,:cpt] = txcs[:cpt]
        tiges_table.yc[id_tige, nbimage,:cpt] = tycs[:cpt]
        tiges_table.theta[id_tige, nbimage,:cpt] = tthetas[:cpt]
        tiges_table.xb1[id_tige, nbimage,:cpt] = txb1[:cpt]
        tiges_table.xb2[id_tige, nbimage,:cpt] = txb2[:cpt]
        tiges_table.yb1[id_tige, nbimage,:cpt] = tyb1[:cpt]
        tiges_table.yb2[id_tige, nbimage,:cpt] = tyb2[:cpt]
        
            
def fast_methode_Olivier(np.ndarray[DTYPE_tf64, ndim=2] image, tiges_table, int id_tige, int nbimage,
                         np.ndarray[DTYPE_t, ndim=1] xi, np.ndarray[DTYPE_t, ndim=1] yi,
                         float pas, int Np, float seuil_coupure=0.2, int show_tige = False,
                         int rayonfilter=True, target=None):
    compute_Olivier(image, tiges_table, id_tige,nbimage,
                    xi, yi, pas, Np, seuil_coupure, show_tige,
                    rayonfilter)