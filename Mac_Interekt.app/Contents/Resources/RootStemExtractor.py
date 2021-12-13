#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:14:54 2015

Petite interface graphique pour traiter les tiges ou les racines

@author: hugo chauvet

Change:
06/09/2019: [Hugo] Fin de la première version compatible HDF5
19/07/2019: [Felix] Ajout du calcul de gamma pour expérience sous clinostat, Hugo ajout de la fonction reset_globals_data
04/05/2019: [Hugo] Correct bugs on export when tige_id_mapper was defined with string names for bases. Allow float in the slide to select sensitivity.
24/09/2018: [Hugo] Remove test dectection (ne marche pas en Thread avec matplotlib!) + Correction bug pour traiter une seule image.
08/06/2018: [Hugo] Correct queue.qsize() bug in osX (car marche pas sur cette plateforme
18/04/2018: [Hugo] correct datetime bug. Set percentdiam to 1.4 in MethodOlivier (previous 0.9). Windows system can now use thread
22/10/2017: [Hugo] Optimisation du positionnement du GUI, utilisation de Tk.grid a la place de Tk.pack
16/10/2015: [Hugo] Ajout de divers options pour la tige (supression etc..) avec menu click droit
                   +Refactorisation de plot_image et de la gestion du global_tige_id_mapper (pour gerer les suppressions)

30/07/2015: [Hugo] Correction bugs pour les racines dans libgravimacro + Ajout visu position de la moyenne de l'ange + Ajout d'options pour le temps des photos
25/05/2015: [Hugo] Ajout des menus pour l'export des moyennes par tiges et pour toutes les tiges + figures
20/05/2015: [Hugo] Première version
"""

import platform
import matplotlib
matplotlib.use('TkAgg')

#Remove zoom key shorcuts
matplotlib.rcParams['keymap.back'] = 'c'
matplotlib.rcParams['keymap.forward'] = 'v'

#from pylab import *
import matplotlib.pylab as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.backend_bases import cursors

from numpy import (array, linspace, arctan2, sqrt, sin, cos, pi,
                   timedelta64, nonzero, arange, hstack, diff,
                   rad2deg, deg2rad, mean, argmin, zeros_like, zeros,
                   ma, cumsum, convolve, ones, exp, log)


from pylab import find
from scipy.optimize import fmin
import pandas as pd
#pd.options.display.mpl_style = 'default'

import cPickle as pkl
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.widgets import RectangleSelector
from new_libgravimacro import (save_results, load_results,
                               traite_tiges2, traite_tige2,
                               plot_sequence,
                               convert_angle, get_photo_time, Image,
                               compute_pattern_motion, ProcessImages,
                               get_tiges_lines)

import rootstem_hdf5_store as h5store

from Extrawidgets import DraggableLines

import sys, os
if sys.version_info[0] < 3:
    import Tkinter as Tk, tkFileDialog, tkMessageBox
else:
    import tkinter as Tk, tkFileDialog, tkMessageBox

import re
finddigits = re.compile(r'\d+?')
from ttk import Style, Button, Frame, Progressbar, Entry, Scale
from threading import Thread
import Queue

__version__ = '06092019'

########################## GLOBAL DATA ########################################
tiges_data = None
img_object = None
text_object = None
tiges_plot_object = []
base_pts_out = None
base_dir_path = None
cur_image = None
cur_tige = None
toptige = None
add_tige = False
nbclick = 0
base_tiges = []
btige_plt = []
btige_text = []
btige_arrow = []
tiges_colors = None
dtphoto = []
thread_process = None  # To store the thread that run image processing
infos_traitement = None
old_step = 0
add_dist_draw = False
dist_measure_pts = []
pixel_distance = None
cm_distance = None
tk_list_images = None # Contient l'objet TK listbox qui accueil les noms des images
tk_toplevel_listimages = None # Contient la fenetre TK qui contient la list des images
hdf5file = None #Contient l'emplacement du fichier hdf5 qui contient les données
PAS_TRAITEMENT = 0.3
tiges_names = [] #Contient un nom pour les tiges 

def reset_graph_data():
    global img_object, text_object, tiges_plot_object, End_point_plot
    global btige_plt, btige_text, tiges_colors, btige_arrow

    img_object = None
    text_object = None
    tiges_plot_object = []
    btige_plt = []
    btige_text = []
    btige_arrow = []
    End_point_plot = {}

    
def reset_globals_data():
    """
    Fonction pour remettre à zéro les variables globales, ala Fortran

    utile lors du chargement d'un nouveau fichier
    """
    print("Remise à zéro des données")
    
    global data_out, text_object, tige_plot_object, base_pts_out, base_dir_path
    global cur_image, cur_tige, toptige, add_tige, btige_arrow
    global nbclick, base_tiges, btige_plt, btige_text, tiges_colors
    global dtphoto, local_photo_path, thread_process, infos_traitement
    global old_step, add_dist_draw, dist_measure_pts, pixel_distance
    global cm_distance, tk_list_images, tk_toplevel_listimages
    global hdf5file, tiges_data, tiges_names
    
    # Reset des dicos
    # RAZ des autres valeurs
    data_out = None
    img_object = None
    text_object = None
    tige_plot_object = None
    base_pts_out = None
    base_dir_path = None
    cur_image=None
    cur_tige = None
    toptige = None
    add_tige = False
    nbclick = 0
    base_tiges = []
    btige_plt = []
    btige_text = []
    btige_arrow = []
    tiges_colors = None
    dtphoto = []
    local_photo_path = False  # If true the photo path is removed
    thread_process = None  # To store the thread that run image processing
    infos_traitement = None
    old_step = 0
    add_dist_draw = False
    dist_measure_pts = []
    pixel_distance = None
    cm_distance = None
    tk_list_images = None
    hdf5file = None
    tiges_data = None
    tiges_names = []
    
    # On ferme la liste des images si elle est ouverte 
    if tk_toplevel_listimages:
        tk_toplevel_listimages.destroy()
        tk_toplevel_listimages = None
        
###############################################################################

######################## CONVERT OLD PKL TO HDF5 ##############################
def convert_pkl_to_hdf(pklfile, hdf5file):
    """
    Fonction pour convertir le vieux format .pkl en hdf5
    """
    global tiges_data
    
    base_dir_path =  os.path.dirname(pklfile)+'/'
    
    display_mpl_message('Ancien format de données: CONVERSION (rootstem_data.pkl -> rootstem_data.h5)\nChargement des résultats',
                        ax, canvas)

    # On charge le fichier pkl
    data_out = load_results(pklfile)

    # On converti les images (avec deux images réduites)
    display_mpl_message('Ancien format de données: CONVERSION (rootstem_data.pkl -> rootstem_data.h5)\nConversion des images...', ax, canvas)

    Nimgs = len(data_out['tiges_info'])
    for tps_step in data_out['tiges_info']:
        # Test si les images sont disponibles à partir du chemin
        # d'origine ou dans le même dossier que le fichier
        # RootStemData.pkl
        if os.path.exists(tps_step['imgname']):
            img_path = tps_step['imgname']
        else:
            img_path= base_dir_path+os.path.basename(tps_step['imgname'])

        # On affiche l'état d'avancement
        display_mpl_message('Ancien format de données: CONVERSION (rootstem_data.pkl -> rootstem_data.h5)\nConversion des images... (%i/%i)' % (tps_step['iimg'], Nimgs),
                            ax, canvas)
        # On augmente la barre de progression du GUI
        dstep = 1/float(Nimgs) * 100.
        #print(dstep)
        prog_bar.step(dstep)
        prog_bar.update_idletasks()
        root.update()

        # On sauvagarde la photo dans le fichier hdf5, on fait garde
        # que le niveau 0 et 2

        # On sauvegarde les valeurs des résolutions de la pyramide
        # juste pour le dernier pas de temps
        store_resolutions = False
        if tps_step['iimg'] == Nimgs-1:
            store_resolutions = True
            
        h5store.image_to_hdf(hdf5file, img_path,
                             tps_step['iimg'],
                             max_pyramid_layers=2,
                             save_resolutions=store_resolutions,
                             selected_resolutions=[0,2])



    # Remise a zero de la progress bar
    prog_bar.step(0.0)
    prog_bar.update_idletasks()
    root.update()

    # Chargement des points de base
    tiges_data = data_out['tiges_data']
    if 'pts_base' in data_out:
        base_tiges = data_out['pts_base']
    else:
        base_tiges = [[(tiges_data.xc[ti,0,0], tiges_data.yc[ti,0,0])]*2 for ti in range(len(tiges_data.yc[:,0,0]))]

    #chargement des numeros des tiges
    if os.path.isfile(base_dir_path+'tige_id_map.pkl'):
        with open(base_dir_path+'tige_id_map.pkl', 'rb') as f:
            datapkl = pkl.load(f)
            tige_id_mapper = datapkl['tige_id_mapper']

            # Cherche si il y a d'autre données du post-processing a
            # charger Leur nom de variable est le même que la clef du
            # dico, on utilise la fonction eval pour créer la variable
            # Expl: scale_cmpix = datapkl['scale_cmpix']
            for dname in ('scale_cmpix', 'L_data', 'G_data', 'B_data', 'Beta_data',
                          'gamma_data', 'End_point_data', 'Tiges_seuil_offset'):
                
                if dname in datapkl:
                    exec("%s = datapkl['%s']" % (dname, dname))
                else:
                    #Creation d'un dico vide
                    exec("%s = {}" % (dname))
    else:
        tige_id_mapper = {}            

    # Creation des dossiers des tiges et du tige_id_mapper si il n'a
    # pas été definit dans le fichier tige_id_map.pkl
    for it in range(len(base_tiges)):
        h5store.create_tige_in_hdf(hdf5file, it)
        if it not in tige_id_mapper:
            tige_id_mapper[it] = it + 1


    # On lance le postrocessing des données des squelette des tiges
    # pour les enregistrer ensuite dans le fichier hdf5
    Traite_data(save_to_hdf5=True)

    # Pour l'échelle de l'image
    scale_cmpix = None
    
    # Chargement des données du traitement dans le fichier hdf5
    for id_tige in range(len(base_tiges)):
        # Est ce qu'il y a un nom pour cette tige
        if id_tige in tige_id_mapper:
            tige_nom = tige_id_mapper[id_tige]
        else:
            tige_nom = ""

        display_mpl_message('Ancien format de données: CONVERSION (rootstem_data.pkl -> rootstem_data.h5)\nConversion des données des squelettes... (tige %i/%i)' % (id_tige+1, len(base_tiges)),
                            ax, canvas)
        h5store.tige_to_hdf5(hdf5file, id_tige, tige_nom, base_tiges[id_tige],
                             tiges_data.xc[id_tige], tiges_data.yc[id_tige],
                             tiges_data.theta[id_tige], tiges_data.diam[id_tige],
                             tiges_data.xb1[id_tige], tiges_data.yb1[id_tige],
                             tiges_data.xb2[id_tige], tiges_data.yb2[id_tige])


        # Enregistrement des données comme dans fonction l'ancienne
        # fonction save_tige_idmapper() On sauve aussi cet ancien
        # tige_id_mapper dans le champ 'name' de la tige qui est plus
        # logique et qui serra dorénavant utilisé
        postprocess_data = {'OLD_tige_id_mapper': tige_id_mapper[id_tige]}

        h5store.save_tige_name(hdf5file, id_tige, str(tige_id_mapper[id_tige]))

        # On boucle sur les autres données du post-processing leur nom
        # est le même que le nom de la variable
        for dname in ('L_data', 'G_data', 'B_data', 'Beta_data',
                      'gamma_data', 'End_point_data', 'Tiges_seuil_offset'):
            try:
                postprocess_data[dname] = eval('%s[%i]' % (dname, id_tige))
            except Exception as e:
                print('No postprocessing data %s for tige %i' % (dname, id_tige))
                print(e)

            # On enregistre la valeur de l'échelle pour la sauvegarder
            # dans la H5 une seule fois à la fin
            if 'scale_cmpix' in postprocess_data:
                scale_cmpix = postprocess_data['scale_cmpix']

        # Cas un peu particulier de B_data qui contient des clefs
        # avec des '/' ce qui fait des sous-dossiers dans le
        # fichier h5, il faut donc les changer
        if 'B_data' in postprocess_data:
            if 'Lc (fit Log(A/A0))' in postprocess_data['B_data']:
                try:
                    postprocess_data['B_data']['Lc (fit Log(A over A0))'] = postprocess_data['B_data'].pop('Lc (fit Log(A/A0))')
                except:
                    print('Pas capable de convertir la clef "Lc (fit Log(A/A0))"')

            if 'Lc (fit exp(A/A0))' in postprocess_data['B_data']:
                try:
                    postprocess_data['B_data']['Lc (fit exp(A over A0))'] = postprocess_data['B_data'].pop('Lc (fit exp(A/A0))')
                except:
                    print('Pas capable de convertir la clef "Lc (fit exp(A/A0))"')
                
        h5store.save_tige_dict_to_hdf(hdf5file, id_tige, postprocess_data)

    # Sauvegarde de l'échelle
    h5store.save_pixelscale(hdf5file, scale_cmpix)
    
def load_hdf5_tiges(hdf5file):
    """
    Fonction pour recharger toutes les tiges depuis le fichier hdf5
    vers les variables globales de rootstem

    load_data: permet de charger en mémoire les données tiges_data et
               celle du post_processing pour afficher les profils des
               tiges voir la fonction Load_postprocessed_data()
    """
    global tiges_data, base_tiges, btige_plt, btige_text, tiges_names
    global tiges_plot_object
    
    # On trouve combien on a de tiges traitées
    Ntiges = h5store.get_number_of_tiges(hdf5file)

    # On crée les listes pour stoker les textes et les plots
    # pour chaque tiges
    btige_plt = [None]*Ntiges
    btige_text = [None]*Ntiges
    tiges_plot_object = [None]*Ntiges
    
    # Chargement du nom des tiges
    display_mpl_message('Chargement des données en mémoire', ax, canvas)
    tiges_names = h5store.get_tiges_names(hdf5file)
    base_tiges = h5store.get_tiges_bases(hdf5file)
    #Recharge la colormap pour les tiges ouvertes
    set_tiges_colormap()
    
    #Traitement des données
    print(u"Chargement des données du fichier h5 dans la mémoire")    
    tiges_data = h5store.get_tigesmanager(hdf5file)
            
def load_hdf5_file(hdf5file):
    """
    Fonction pour charger un fichier hdf5 dans rootstemextractor
    """
    global dtphoto
    
    # On charge les données des tiges
    load_hdf5_tiges(hdf5file)
    
    # Doit on charger le temps des photos 
    if get_photo_datetime.get():
        print('Chargement des photos')
        dtphoto = h5store.get_images_datetimes(hdf5file)
        # On regarde si tous les temps sont des datetime sinon on
        # prends le numéro de la photo
        for t in dtphoto:
            if t is None:
                print("Erreur pas de temps pour toutes les photos, on utilise le numéros des images à la place")
                dtphoto = []
                break
    else:
        print('Pas de temps, on utilise le numéro des photos à la place')
        dtphoto = []

    # Si pas de temps dans les photos on mets l'option dans
    # l'interface graphique à False
    if dtphoto == []:
        get_photo_datetime.set(False)

def get_hdf5_tigeid(h5file, gui_tige_id):
    """
    Fonction qui renvoie l'id sous lequel est enregistrer la tige dans le fichier hdf5
    """

    tiges_ids = h5store.get_tiges_indices(h5file)
    tigeid = tiges_ids[gui_tige_id]

    return tigeid

########################## LOAD FILES #########################################
def _open_files():
    """
    Fonction pour charger un fichier dans rootstem extractor.
    
    Ce fichier peut soit être un fichier .pkl (ancien format de
    rootstem), une liste d'image pour un nouveau traitement, ou un
    fichier hdf5 (le nouveau format de rootstem)
    """
    global base_dir_path, data_out, imgs_out, base_tiges
    global btige_plt, btige_text, dtphoto, cidkey
    global hdf5file, tiges_data
    
    reset_graph_data()

    #TODO: Bug mac version TK and wild-cards
    #OLD, ('Images', '*.jpg,*.JPG'), ('Projet', '*.pkl')
    #ftypes = [('all files', '.*'), ('Images', '*.jpg,*.JPG'), ('Projet', '*.pkl')]

    #todo  filetypes=ftypes
    files = tkFileDialog.askopenfilenames(parent=root, title='Choisir les images a traiter')

    if files != '' and len(files) > 0:

        # On fait une RAZ toutes les données globales dans rootstem
        reset_globals_data()
        
        #base_dir_path = os.path.abspath(files[0]).replace(os.path.basename(files[0]),'')
        base_dir_path =  os.path.dirname(files[0])+'/'
        base_dir_path = base_dir_path.encode(sys.getfilesystemencoding()) # ENCODING POUR LES NOMS de dossiers avec de l'utf8
        #Test si c'est un fichier de traitement ou des images qui sont chargées

        # Si c'est un fichier en pkl (ancienne version, il faut le convertir en hdf5)
        if '.pkl' in files[0]:

            output_hdf5_file = base_dir_path + 'rootstem_data.h5'
            hdf5file = output_hdf5_file

            # A t on déjà un fichier h5
            process = True
            if os.path.exists(hdf5file):
                # On demande si on vuet continuer en supprimant le fichier 
                resp = tkMessageBox.askokcancel('Un fichier de traitement rootstem_data.h5 existe',
                                                "Le fichier rootstem_data.h5 existe déjà voulez vous continuer?\nCela supprimera le fichier rootstem_data.h5 existant")
                
                if resp:
                    os.unlink(hdf5file)
                else:
                    process = False

            if process:
                # On lance la conversion
                convert_pkl_to_hdf(files[0], output_hdf5_file)
            
                # On charge le fichier créé
                load_hdf5_file(hdf5file)

        elif '.h5' in files[0]:

            # On enregistre le chemin vers le fichier hdf5file
            hdf5file = files[0]
            
            # On charge le fichier 
            load_hdf5_file(hdf5file)
            
        else:

            # Nom du fichier
            hdf5file = base_dir_path + 'rootstem_data.h5'
            
            # Si on ouvre des images, il faut verifier que le fichier
            # h5 n'existe pas sinon on le supprime en mettant un
            # message d'avertissement.
            process = True
            if os.path.exists(hdf5file):
                # On demande si on vuet continuer en supprimant le fichier 
                resp = tkMessageBox.askokcancel('Un fichier de traitement existe',
                                                "Le fichier rootstem_data.h5 existe déjà voulez vous continuer?\nCela supprimera le fichier rootstem_data.h5 existant")
                
                if resp:
                    os.unlink(hdf5file)
                else:
                    process = False
                    
            if process:
                # Chargement des nom des images

                # ENCODING DUMMY
                files_to_process = [f.encode(sys.getfilesystemencoding()) for f in files]
                #print files_to_process, files
                #f.encode(sys.getfilesystemencoding())

                #Try to sort images with their numbers
                try:
                    files_to_process = sorted(files_to_process, key=lambda x: int(''.join(finddigits.findall(x.split('/')[-1]))) )
                except:
                    print(u"Pas de trie des photos...car pas de numéros dans le nom des images!")

                # Boucle sur les fichier image pour les ajouter dans un nouveau fichier HDF           
                Nimgs = len(files_to_process)
                for i, f in enumerate(files_to_process):
                    # On affiche l'état d'avancement
                    display_mpl_message('Création du fichier rootstem_data.h5\ndans le dossier %s\nConversion des images... (%i/%i)' % (base_dir_path, i, Nimgs),
                                        ax, canvas)

                    # On augmente la barre de progression du GUI
                    dstep = 1/float(Nimgs) * 100.
                    #print(dstep)
                    prog_bar.step(dstep)
                    prog_bar.update_idletasks()
                    root.update()
                
                    # On sauvegarde la pyramide de résolution juste pour le
                    # dernier pas de temps
                    store_resolutions = False
                    if i == Nimgs-1:
                        store_resolutions = True
            
                    h5store.image_to_hdf(hdf5file, f, i,
                                         max_pyramid_layers=2,
                                         save_resolutions=store_resolutions,
                                         selected_resolutions=[0,2])


                # Chargement du temps a partir du fichier hdf5
                dtphoto = h5store.get_images_datetimes(hdf5file)
            
    change_button_state()
    plot_image(0, force_clear=True)


    #Restore focus on the current canvas
    canvas.get_tk_widget().focus_force()

def change_button_state():

    if len(base_tiges) > 0:
        for bt in [button_supr_all_tige, button_traiter]:
            bt.config(state=Tk.NORMAL)
    else:
        for bt in [button_supr_all_tige, button_traiter]:
            bt.config(state=Tk.DISABLED)

    if len(h5store.get_images_names(hdf5file)) > 0:
        for bt in [button_addtige, button_listimages]:
            bt.config(state=Tk.NORMAL)
    else:
        for bt in [button_addtige, button_supr_all_tige, button_traiter, button_listimages]:
            bt.config(state=Tk.DISABLED)


def display_mpl_message(message, axe, img_canvas, msg_color='red'):
    """
    Fonction pour faire apparaître un message au centre du canvas matplotlib

    Paramètres
    ----------

    message, string:
        Message a afficher

    axe, matplotlib ax object:
        Objet de type matplotlib.axes

    img_canvas, matplotlib canvas object:
        Objet renvoyant le canvas de la figure
        
    msg_color, string optional:
        Permet de définir la couleur du message affiché

    Exemple:
    --------

    fig = figure()
    ax = fig.add_subplot(111)

    display_mpl_message('This is my message', ax, fig.canvas)
    """
    
    axe.clear()
    axe.axis("off")
    axe.text(0.5, 0.5, message.decode('utf8'),
            ha='center', va='center', color=msg_color,
            transform=axe.transAxes)
    img_canvas.draw()

# La variable globale update_imgid permet de stocker le after de
# tkinter qui met a jour la figure en fullresolution et de pouvoir
# l'annuler si on change l'image trop vite.
update_imgid = None
def plot_image_fullres():
    """
    Affiche l'image en pleine résolution. Cette fonction est lancée par
    la méthode after de tkinter pour améliorer la résolution de l'image
    quand on reste dessus.
    """
    global img_object, update_imgid

    # On charge l'image en basse résolution
    if cur_image is not None:
        plot_image(cur_image, keep_zoom=True, image_resolution=0)
        
    # Remet l'id pour tk after a None
    update_imgid = None

#global pour sauvegarde la résolution utilisée cela permet de faire
#suivre les xlims ylims de la figure correctement
used_scale = 1
def plot_image(img_num, keep_zoom=False, force_clear=False, image_resolution=1):
    global cur_image, btige_plt, btige_text, img_object, text_object
    global update_imgid, used_scale

    # Doit on annuler la mise a jour de l'image fullresolution
    if update_imgid is not None:
        root.after_cancel(update_imgid)
        update_imgid = None
        
    # Met a jour la variable globale cur_image
    cur_image = img_num

    # Doit on garder le zoom en mémoire pour le restaurer à la fin
    if keep_zoom:
        oldxlims = ax.get_xlim()
        oldylims = ax.get_ylim()

    # Doit on tout nétoyer
    if force_clear:
        ax.clear()
        
        #Reset btige
        btige_plt = [None] * len(base_tiges)
        btige_text = [None] * len(base_tiges)

        
    if img_num is None:
        ax.axis('off')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        if text_object is None:
            text_object = ax.text(0.5,0.5,"Charger des images \nou\n un fichier de resultat (rootstem_data.pkl)",ha='center',va='center')
            fig.tight_layout()
        else:
            text_object.set_text("Charger des images \nou\n un fichier de resultat (rootstem_data.pkl)")

    else:
        ax.axis('on')

        # On charge l'image en petite résolution
        imtmp = h5store.open_hdf5_image(hdf5file, img_num,
                                        resolution=image_resolution)

        scalefactors = h5store.get_image_scale(hdf5file,
                                               resolution=image_resolution)        
        # Changer le titre de la fenetre principal pour afficher la photo
        try:
            root.wm_title("RootStemExtractor | %s" % (h5store.get_images_names(hdf5file,
                                                                               img_num)))
        except Exception as e:
            print('Erreur dans le changement de nom de la fenêtre')
            print(e)

        # Mise a jour de la position sélectionnée dans la liste des
        # images si elle existe (si la fenetre "tk_toplevel_listimages"
        # est ouverte)
        if tk_list_images:
            tk_list_images.selection_clear(0, Tk.END)
            tk_list_images.selection_set(img_num)
            tk_list_images.see(img_num)
            
        if img_object is None:
            if imtmp.shape < 3:
                 img_object = ax.imshow(imtmp, cmap=mpl.cm.gray)
            else:
                img_object = ax.imshow(imtmp)
        else:
            #print shape(imtmp)
            img_object.set_data(imtmp)
            # Mise a jour de l'extent de l'imshow
            img_object.set_extent((0, imtmp.shape[1], imtmp.shape[0], 0))

        #Plot des bases de tiges
        image_scale = 1/float(scalefactors[0])
        plot_basetiges(ratio=image_scale)

        if image_resolution > 0:
            tige_plot_decimation = 100 #On decime le nombre de points sur les tiges pour augmenter la vitesse d'affichage
        else:
            tige_plot_decimation = 10 # Quand on est en fullresolution on decime moins les tiges
            
        if tiges_data is not None:
            #Plot des tiges traitée
            for ti, tige_xyplot in enumerate(tiges_plot_object):
                # On teste d'abord si il y a des données 
                tigeh5id = get_hdf5_tigeid(hdf5file, ti)
                
                if h5store.is_data(hdf5file, 'xc', tigeh5id):
                    
                    # Creation des graphiques
                    if tige_xyplot is None:
                        # Les tiges sont tracé avec une décimation définit
                        # avec le scalefactor
                        try:
                            tmp, = ax.plot(tiges_data.xc[ti,int(img_num),::tige_plot_decimation].T*image_scale,
                                           tiges_data.yc[ti,int(img_num),::tige_plot_decimation].T*image_scale,
                                           lw=3, picker=5, label="%i"%(ti),
                                           color=tiges_colors[ti])
                            tiges_plot_object[ti] = tmp
                        except Exception as e:
                            print(u'Erreur de chargement des données pour la tige %s en position %i du gui'%(tiges_names[ti], ti))
                            print(e)

                    # Mise a jour des graphiques
                    else:
                        try:
                            tiges_plot_object[ti].set_data(tiges_data.xc[ti,int(img_num),::tige_plot_decimation].T*image_scale,
                                                           tiges_data.yc[ti,int(img_num),::tige_plot_decimation].T*image_scale)
                        except Exception as e:
                            print(u'Erreur de mise à jour des données pour la tige %s en position %i du gui'%(tiges_names[ti], ti))
                            print(e)
        else:
            print("Aucune tige n'a de données")                
    

        if keep_zoom:
            #print(oldxlims, oldylims)
            # Ici on gère si c'est un grandissement
            if used_scale > scalefactors[0]:
                oldxlims = [l*used_scale for l in oldxlims]
                oldylims = [l*used_scale for l in oldylims]

            # Quand c'est un rétrécissement
            if used_scale < scalefactors[0]:
                oldxlims = [l/float(scalefactors[0]) for l in oldxlims]
                oldylims = [l/float(scalefactors[0]) for l in oldylims]
            
            #print(used_scale, scalefactors[0])
            #print(oldxlims, oldylims)
            
            ax.set_xlim(oldxlims)
            ax.set_ylim(oldylims)

    canvas.draw()
    print(u'fin de mise à jour de la figure à la résolution %i' % image_resolution)

    # Mise a jour de la memoire de la résolution utilisée
    if img_num is not None:
        used_scale = scalefactors[0]
    # On lance un after si pas de changement d'image et que la
    # résolution est une résolution plus faible que l'image originale
    if img_num is not None and image_resolution > 0:
        update_imgid = root.after(350, plot_image_fullres)
    
########################### GESTION GRAPHIQUE DES TIGES #######################
def _addtige():
    """
    Fonction pour activer l'ajout d'une tige avec le gui (voir la
    gestion des clics plus bas)
    """
    global add_tige, nbclick, btige_plt, btige_text
    add_tige = True
    nbclick = 0
    change_button_state()


def _supr_all_tiges():
    """
    Supression de toutes les tiges dans le fichier hdf5
    """
    reset_graph_data()

    tiges_ids_hdf = h5store.get_tiges_indices(hdf5file)

    for idt in tiges_ids_hdf:
        h5store.delete_tige(hdf5file, idt)

    # Recharge les données depuis le fichier hdf
    load_hdf5_tiges(hdf5file)
    #Replot the current image
    plot_image(cur_image, force_clear=True)
    change_button_state()

def set_tiges_colormap():
    global tiges_colors
    #Fonction pour construire le vecteur des couleurs pour les tiges
    if len(base_tiges) > 20:
        s = len(base_tiges)+1
    else:
        s = 20
        
    try:
        tiges_colors = mpl.cm.tab20(linspace(0, 1, s))
    except Exception as e:
        print(e, "Mets à jour Matplotlib !")
        tiges_colors = mpl.cm.hsv(linspace(0, 1, s))

def plot_basetiges(force_redraw=False, ratio=1):
    """
    Trace les bases des tiges sur le graphique avec la ligne du centre
    de la tige si le traitement a été effectué.

    force_redraw: Si True, On force matplotlib a retracer la figure 
    ratio: avec quelle ratio d'image doit on dessiner les tiges
           (permet de tracer avec des images dont la résolution a été
           diminuée)
    """
    global btige_plt, btige_text, btige_arrow

    print('plot base tige')
    
    oldxlim = ax.get_xlim()
    oldylim = ax.get_ylim()
    to_remove = []

    # On efface les arrow des tiges
    for a in btige_arrow:
        try:
            a.remove()
        except Exception as e:
            print(u'Erreur lors de la suppression de la flèche de la base')
            print(e)
    # Reinitialise la list des fleches
    btige_arrow = []
    
    for i in range(len(base_tiges)):
        #print('Ajout de la base %i' %i)
        #check tige name
        try:
            tname = tiges_names[i]
            base = base_tiges[i]
        except:
            print('No base data for %i'%i)
            base = []

        if len(base) > 1:
            if base[0][0] < base[1][0]:
                #Vers le haut
                symb='r-'
            else:
                symb='m-'

            if btige_plt[i] is None:
                btige_plt[i], = ax.plot([base[0][0]*ratio,base[1][0]*ratio],
                                        [base[0][1]*ratio,base[1][1]*ratio],
                                        symb, label='base_%i'%i, lw=1.1, picker=5)
            else:
                btige_plt[i].set_data([base[0][0]*ratio,base[1][0]*ratio],
                                      [base[0][1]*ratio,base[1][1]*ratio])

            #Dessin de la normale
            theta = arctan2(base[1][1]-base[0][1], base[1][0]-base[0][0])
            L = 0.25 * sqrt((base[1][1]-base[0][1])**2 + (base[1][0]-base[0][0])**2) * ratio #Taille en pixel de la normale
            xc = 0.5*(base[1][0]+base[0][0])*ratio
            yc = 0.5*(base[1][1]+base[0][1])*ratio
            xn = L * cos(theta-pi/2.)
            yn = L * sin(theta-pi/2.)
            ar = ax.arrow(xc, yc, xn, yn, color=symb[0], length_includes_head=True, head_width=2)
            btige_arrow += [ar]

            if btige_text[i] is None:
                btige_text[i] = ax.text(base[0][0]*ratio, base[0][1]*ratio, '%s'%str(tname), color='r')
            else:
                btige_text[i].set_text('%s'%str(tname))
                btige_text[i].set_x(base[0][0]*ratio)
                btige_text[i].set_y(base[0][1]*ratio)

        ax.set_xlim(oldxlim)
        ax.set_ylim(oldylim)

        
    if force_redraw:
        canvas.draw()

###############################################################################

############################ Affiche la liste des images ######################
def onChangeImage(event):
    try:
        sender=event.widget
        idx = sender.curselection()
        #print("idx %i"%idx)
        plot_image(idx[0], keep_zoom=True)
    except Exception as e:
        print(u"Erreur de chargement !!!")
        print(e)

def show_image_list():
    global tk_list_images, tk_toplevel_listimages

    if tk_toplevel_listimages is None:
        tk_toplevel_listimages = Tk.Toplevel(master=root)
        tk_toplevel_listimages.title("Liste des images ouvertes")


        topsbar = Tk.Scrollbar(tk_toplevel_listimages, orient=Tk.VERTICAL)
        listb = Tk.Listbox(master=tk_toplevel_listimages,
                           yscrollcommand=topsbar.set)
        topsbar.config(command=listb.yview)
        topsbar.pack(side=Tk.RIGHT, fill=Tk.Y)

        for imgname in h5store.get_images_names(hdf5file):
            listb.insert(Tk.END, imgname)

        # Selection de l'image que l'on regarde
        if cur_image:
            listb.selection_set(cur_image)
            listb.see(cur_image)
        else:
            listb.selection_set(0)
            listb.see(0)

        listb.bind("<<ListboxSelect>>", onChangeImage)
        listb.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
        tk_list_images = listb

        # Signal pour déclencher la fonction qui ferme proprement la
        # liste d'image en gérant les globals
        tk_toplevel_listimages.protocol("WM_DELETE_WINDOW", destroy_list_images)
        
    else:
        # Si la fenêtre existe déjà mais est cachée on la ramène devant
        tk_toplevel_listimages.lift()
        
def destroy_list_images():
    """
    Fonction pour fermer correctement la fenêtre de la liste des images
    """
    global tk_list_images, tk_toplevel_listimages

    # On détruit la fenetre 
    tk_toplevel_listimages.destroy()

    # On remet les variables global à None (Fortran Style)
    tk_list_images = None
    tk_toplevel_listimages = None
    
###############################################################################


########################## Pour le Traitement #################################
def _export_to_csv():
    """
    Fonction pour exporter la serie temporelle pour les tiges soit
    l'évolution de l'angle moyen au bout et de la taille au cours du
    temps
    """

    if len(base_tiges) > 0:
        proposed_filename = "Serie_tempotelle_AngleBout_et_taille.csv"
        outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                     filetypes=[("Comma Separated Value, csv","*.csv")],
                                                     title="Export serie temporelle",
                                                     initialfile=proposed_filename,
                                                     initialdir=base_dir_path)

    if len(outfileName) > 0:

        # Creation du tableau Pandas pour stocquer les données
        data_out = pd.DataFrame(columns=['tige', 'angle', 'temps',
                                         'taille', 'rayon', 'angle_0_360'])

        # Récupérations des id des tiges dans le fichier h5
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)
        tiges_names = h5store.get_tiges_names(hdf5file)
        pictures_names = h5store.get_images_names(hdf5file)
        
        # Récupère le temps
        if get_photo_datetime.get() and dtphoto != []:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))
            
        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_taille, tige_angle, tige_zone, _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tname = tiges_names[i]
                
            data_tmp = pd.DataFrame({'tige': tname, 'angle': tige_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'angle_0_360': convert_angle(tige_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)

        #Add some usefull data
        #Ntige = data_out.tige.unique()
        output = []
        #print Ntige
        for tige, datatige in data_out.groupby('tige'):
            # print(dataout.tige)
    
            #Compute dt (min)
            if get_photo_datetime.get():
                dtps = (datatige['temps']-datatige['temps'].min())/timedelta64(1,'m')
                datatige.loc[:,'dt (min)'] = dtps

            datatige.loc[:,'pictures name'] = pictures_names

            # Trouve la position de la tige dans les données du GUI
            itige = tiges_names.index(tige)
            
            datatige.loc[:,'x base (pix)'] = [tiges_data.xc[itige,:,0].mean()]*len(datatige.index)
            output += [datatige]

        dataout = pd.concat(output)
        print(dataout.head())
        #print datatige.tail()
        dataout.to_csv(outfileName, index=False)

def _export_xytemps_to_csv():
    """
    Export du squelette des tiges/racines au cours du temps sous
    format csv, pour un pas de temps on aura le xy complet (soit pour
    chaque abscisse curviligne).
    """

    if len(base_tiges) > 0:
        proposed_filename = "Squelette.csv"
        outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                  filetypes=[("Comma Separated Value, csv","*.csv")],
                                                  title="Export serie temporelle",
                                                  initialfile=proposed_filename,
                                                  initialdir=base_dir_path)

    if len(outfileName) > 0:
        # Creation du tableau Pandas pour stocquer les données
        data_out = pd.DataFrame(columns=['tige', 'angle', 'temps',
                                         'taille', 'rayon', 'x', 'y', 'nom photo',
                                         'angle_0_360'])

        # Récupérations des id des tiges dans le fichier h5
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)
        tiges_names = h5store.get_tiges_names(hdf5file)
        pictures_names = h5store.get_images_names(hdf5file)
        
        # Récupère le temps
        if get_photo_datetime.get() and dtphoto != []:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))

        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_taille, tige_angle, tige_zone, _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tname = tiges_names[i]
                
            data_tmp = pd.DataFrame({'tige': tname, 'angle': tige_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'x': tige_x.tolist(),
                                     'y': tige_y.tolist(),
                                     'nom photo': pictures_names,
                                     'angle_0_360': convert_angle(tige_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)

        print(data_out.head())
        print(u"Enregistré dans %s" % outfileName)
        data_out.to_csv(outfileName, index=False)

def _export_mean_to_csv():
    """
    Fonction qui exporte la courbe moyennée pour toutes les tiges de
    l'angle et taille en fonction du temps
    """

    if len(base_tiges) > 0:
        proposed_filename = "Serie_tempotelle_moyenne.csv"
        outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                  filetypes=[("Comma Separated Value, csv","*.csv")],
                                                  title="Export serie temporelle",
                                                  initialfile=proposed_filename,
                                                  initialdir=base_dir_path)

    if len(outfileName) > 0:

         # Creation du tableau Pandas pour stocquer les données
        data_out = pd.DataFrame(columns=['tige', 'angle', 'temps',
                                         'taille', 'rayon', 'angle_0_360'])

        # Récupérations des id des tiges dans le fichier h5
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)
        tiges_names = h5store.get_tiges_names(hdf5file)
        pictures_names = h5store.get_images_names(hdf5file)
        
        # Récupère le temps
        if get_photo_datetime.get() and dtphoto != []:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))
            
        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_taille, tige_angle, tige_zone, _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tname = tiges_names[i]
                
            data_tmp = pd.DataFrame({'tige': tname, 'angle': tige_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'angle_0_360': convert_angle(tige_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)
            

        #Creation de la moyenne
        datamoy = data_out.groupby('temps').mean()
        datamoy['temps'] = data_out.temps.unique()
        datamoy['tige'] = ['%s->%s'%(str(data_out['tige'].min()),
                                     str(data_out['tige'].max()))]*len(datamoy['temps'])
        
        #Convert to timedelta in minute
        if get_photo_datetime.get():
            dtps = (datamoy['temps']-datamoy['temps'][0])/timedelta64(1,'m')
            datamoy['dt (min)'] = dtps

        datamoy['pictures name'] = pictures_names
        print(u"Saved to %s"%outfileName)
        datamoy.to_csv(outfileName, index=False)

def _export_meandata_for_each_tiges():
    """
    Dans un dossier donnée par l'utilisateur on exporte les données
    taille et angle au cours de temps avec un fichier .csv par
    tiges/racines
    """

    #Ask for a directory where to save all files
    outdir = tkFileDialog.askdirectory(title=u"Choisir un répertoire pour sauvegarder les tiges")
    if len(outdir) > 0:
        # Creation du tableau Pandas pour stocquer les données
        data_out = pd.DataFrame(columns=['tige', 'angle', 'temps',
                                         'taille', 'rayon', 'angle_0_360'])

        # Récupérations des id des tiges dans le fichier h5
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)
        tiges_names = h5store.get_tiges_names(hdf5file)
        pictures_names = h5store.get_images_names(hdf5file)
        
        # Récupère le temps
        if get_photo_datetime.get() and dtphoto != []:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))
            
        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_taille, tige_angle, tige_zone, _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tname = tiges_names[i]
                
            data_tmp = pd.DataFrame({'tige': tname, 'angle': tige_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'angle_0_360': convert_angle(tige_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)
            
        #Loop over tiges
        # Ntige = dataframe.tige.unique()
        for tige, datatige in data_out.groupby('tige'):
            #Compute dt (min)
            if get_photo_datetime.get():
                dtps = (datatige['temps']-datatige['temps'][datatige.index[0]])/timedelta64(1,'m')
                datatige['dt (min)'] = dtps

            datatige['pictures name'] = pictures_names
            # Trouve la position de la tige dans les données du GUI
            itige = tiges_names.index(tige)
            datatige['x base (pix)'] = [tiges_data.xc[itige,:,0].mean()]*len(datatige.index)

            #print datatige.head()
            outfileName = outdir+'/data_mean_for_tige_%s.csv'%str(tige)
            print(u"Sauvegardé dans %s"%outfileName)
            datatige.to_csv(outfileName, index=False)

def update_tk_progress(infos, root):
    global old_step
    if infos is not None:

        im_num = infos['inum']
        old_num = infos['old_inum']
        tot = infos['tot']
        #print im_num, old_num, tot
        msg = "Traitement de l'image %i / %i"%(int(im_num),int(tot))
        root.wm_title("RootStemExtractor | %s"%msg)
        root.update_idletasks()


        #New version with Tkk progressbar
        if im_num != old_step and tot > 1:
            old_step = im_num
            #print(old_step, nstep, nstep-old_step)
            dstep = (im_num-old_num)/float(tot-1) * 100.
            #print(dstep)
            prog_bar.step(dstep)
            prog_bar.update_idletasks()
            root.update()

def plot_progress(**kwargs):
    global infos_traitement
    infos_traitement = kwargs

def none_print(**kwargs):
    output = kwargs

def check_process():
    global root, infos_traitement

    if thread_process.isAlive():
        update_tk_progress(infos_traitement, root)
        root.after(20, check_process)

    else:
        update_tk_progress(infos_traitement, root)
        root.wm_title("RootStemExtractor")
        thread_process.join()
        infos_traitement = None
        try:
            process_data()
        except Exception as e:
            print('Failed to process data')
            print(e)

def process_data():
    global data_out, tiges_data

    #Get Queue result
    data_out = data_out.get()
    # print(data_out.keys())

    #When it's done get the tiges_data from the data from data_out
    tiges_data = data_out['data']['tiges_data']
    
    #On affiche que l'on fait la sauvegarde
    display_mpl_message("Sauvegarde des données dans le fichier .h5", ax, canvas)
    for idt in range(len(base_tiges)):
        # Recup l'id dans le fichier hdf
        idhdf = get_hdf5_tigeid(hdf5file, idt)
        tige_name = h5store.get_tiges_names(hdf5file, idhdf)
        
        h5store.tige_to_hdf5(hdf5file, idhdf, tige_name, base_tiges[idt],
                             tiges_data.xc[idt], tiges_data.yc[idt],
                             tiges_data.theta[idt], tiges_data.diam[idt],
                             tiges_data.xb1[idt], tiges_data.yb1[idt],
                             tiges_data.xb2[idt], tiges_data.yb2[idt])
        
    # On lance le traitement et on le sauvegarde dans le fichier hdf5
    display_mpl_message("Traitement des données extraites", ax, canvas)
    Traite_data(save_to_hdf5=True)

    # Recharge les données des tiges a partir du fichier hdf5
    load_hdf5_tiges(hdf5file)

    #Plot the first image in the list
    plot_image(cur_image, force_clear=True)

    change_button_state()
    prog_bar.stop()

    #Restore focus on the current canvas
    canvas.get_tk_widget().focus_force()

def launch_process():
    """
    Fonction pour lancer le traitement des images chargées dans
    rootstem pour les tiges tracées.
    """
    global data_out, old_step, thread_process, Crops_data


    Crops_data = []
    #Windows OK in thread now
    is_thread = True

    if len(base_tiges) > 0:

        #Create a process to start process image (that can into all processors)
        #is_thread = False
        reset_graph_data()
        data_out = Queue.Queue()

        # On regarde le nombre d'image que l'on a
        Nimgs = h5store.get_number_of_images(hdf5file)

        # Récup les options pour les tiges (on a que le seuil de coupure)
        # Attention cela retourne un dictionnaire avec aa['hdf5_id'] = value
        hdf_tiges_seuil_offset = h5store.get_postprocessing(hdf5file, 'Tiges_seuil_offset')

        # Il faut convertir les id hdf en position des tiges qui correspondent a base_tiges
        Tiges_seuil_offset = {}
        for i in range(len(base_tiges)):
            hdf5_id = get_hdf5_tigeid(hdf5file, i)
            if hdf5_id in hdf_tiges_seuil_offset:
                # On regarde si la valeur enregistrée est pas None et
                # si c'est le cas on met la valeur de l'offset à 0
                tmp_offset = hdf_tiges_seuil_offset[hdf5_id]
                if tmp_offset is None:
                    tmp_offset = 0
                    
                Tiges_seuil_offset[i] = tmp_offset
            else:
                Tiges_seuil_offset[i] = 0
                
        #print('Les offset des tiges')
        #print(Tiges_seuil_offset)
        if Nimgs > 0:
            
            # Doit on faire un pre-processing pour estimer la taille max (en faisant la détection sur la dernière image)
            if Nimgs > 5:
                print('Pre-processing last image (to get maximum plant size)')
                #Preprocessing to guess the max size of the objects and set crop zone for each of them
                pre_data = {}

                pre_data = ProcessImages(file_names=hdf5file,
                                         num_images=[Nimgs-1, Nimgs],
                                         num_tiges=len(base_tiges),
                                         base_points=base_tiges, thread=is_thread,
                                         pas=PAS_TRAITEMENT,
                                         tiges_seuil_offset=Tiges_seuil_offset,
                                         output_function=none_print)


                tiges_x, tiges_y, tiges_tailles, tiges_angles, tiges_lines,  = traite_tiges2(pre_data[0]['tiges_data'],
                                                                                             pas=PAS_TRAITEMENT)
                #print(tiges_tailles/0.3)
                #print(tiges_x.shape)
                max_array_size = tiges_x.shape[2] + 100
                print("Taille maximum pour les itérations %i" % max_array_size)
                
            else:

                max_array_size = 10000

            # Creation du thread principal qui va traiter toutes les images
            thread_process = Thread(name="ImageProcessing",
                                    target=ProcessImages,
                                    kwargs={'file_names':hdf5file,
                                            'num_images':'all',
                                            'num_tiges':len(base_tiges),
                                            'base_points':base_tiges,
                                            'output_function':plot_progress,
                                            #'output_function_args': {'root':root},
                                            'thread':is_thread,
                                            'pas':PAS_TRAITEMENT,
                                            'outputdata':data_out,
                                            'tiges_seuil_offset': Tiges_seuil_offset,
                                            'memory_size': max_array_size,
                                            'crops':Crops_data})

            # On lance le thread
            thread_process.setDaemon(True)
            thread_process.start()

            # Lance la fonction qui va surveiller l'avancement du traitement des images 
            check_process()


def Traite_data(save_to_hdf5=False):
    """Fonction qui lance le post processing pour sortir les variation
    temporelles des données issues du traitement des tiges. Elle
    utilise la fonction traite_tiges2 de new_libgravimacro qui permet
    de lisser les squelettes et renvoie les séries temporelles pour
    l'angle au bout (moyennée sur un certain nombre de points le long
    de la tige) "tiges_angles" (dimension Ntiges x Nimages), la taille
    "tiges_tailles" (dimension Ntiges x Nimages), le squelette (x
    centre et y centre) de la tige lissé ("tiges_x", "tige_y",
    dimension Ntiges x Nimages x abscisse curv), un zip(x,y) contenant
    les squelette ("tiges_lines") et la délimitation de la zone le
    long du profil de la tige ou l'on fait la moyenne de l'angle au
    bout ("tiges_measure_zone", dimension Ntiges x Nimages)
    """
    tiges_x, tiges_y, tiges_tailles, tiges_angles, tiges_lines, tiges_measure_zone = traite_tiges2(tiges_data, pas=PAS_TRAITEMENT,
                                                                                                   return_estimation_zone=True)

    if save_to_hdf5:
        for id_tige in range(len(tiges_x)):
            data_tmp = {"smooth_xc": tiges_x[id_tige],
                        "smooth_yc": tiges_y[id_tige],
                        "taille": tiges_tailles[id_tige],
                        "angle_au_bout": tiges_angles[id_tige],
                        "angle_zone_de_mesure": tiges_measure_zone[id_tige]}

            tigeidhdf = get_hdf5_tigeid(hdf5file, id_tige)
            h5store.save_tige_dict_to_hdf(hdf5file, tigeidhdf,
                                          data_tmp)
    else:
        return tiges_x, tiges_y, tiges_tailles, tiges_angles, tiges_lines, tiges_measure_zone    


def load_postprocessed_data(hdf5file, hdf_tige_id):
    """
    Fonction pour charger les données issue de traite_tiges2 et
    enregistrées dans le fichier hdf5 pour la tige définit par son
    identifiant dans le fichier hdf.
    
    Voir la fonction Traite_data()
    """

    print("Chargement des données de la tige %s depuis le fichier h5" % hdf_tige_id)
    # Recuperation des données dont on a besoin
    tige_x = h5store.get_postprocessing(hdf5file, 'smooth_xc', hdf_tige_id)
    tige_y = h5store.get_postprocessing(hdf5file, 'smooth_yc', hdf_tige_id)
    tige_taille = h5store.get_postprocessing(hdf5file, 'taille', hdf_tige_id)
    tige_angle = h5store.get_postprocessing(hdf5file, 'angle_au_bout', hdf_tige_id)
    tige_measure_zone = h5store.get_postprocessing(hdf5file, 'angle_zone_de_mesure',
                                                   hdf_tige_id).tolist()

    print("Chargement fini!")

    # Masque les données invalide qui ont une valeur de -30000
    badval = -30000
    tige_x = ma.masked_equal(tige_x, badval)
    tige_y = ma.masked_equal(tige_y, badval)
    tige_taille = ma.masked_equal(tige_taille, badval)
    tige_angle = ma.masked_equal(tige_angle, badval)

    # Creation d'une liste de la ligne du centre de la tige compatible
    # avec matplotlib LineCollection pour afficher le squelette de la
    # tige au cours du temps
    # Version très lente l'ancienne version
    # tige_lines = [zip(tige_x[t], tige_y[t]) for t in range(len(tige_x))]

    # Version beaucoup plus rapide
    tige_lines = ma.dstack((tige_x, tige_y))
    
    return tige_x, tige_y, tige_taille, tige_angle, tige_measure_zone, tige_lines

###############################################################################

###################### Gestion des actions du menu pour les tiges individuelles ####

def remove_tige():
    """
    Fonction pour supprimer une tige qui est sélectionnée par la
    valeur contenue dans la variable globale cur_tige.

    Étapes de la suppression:
    1- on trouve l'id de la tige dans le fichier hdf5 (car elle
       peuvent avoir des ids discontinue)
    2- on supprime les données de cette tige dans le fichier hdf5 
    3- on re-charge à partir du fichier hdf5 les données des variables
       globales: tiges_data, tige_id_mapper, base_tiges
    4- on fait un refresh des plots
    """

    global floatmenuisopen
    # Force la fermeture du menu popup dans tk
    floatmenuisopen = False
    
    # Récupération des ids dans le fichier hdf5
    tigeid = get_hdf5_tigeid(hdf5file, cur_tige)
    tname = h5store.get_tiges_names(hdf5file, tigeid)
    
    print('Suppresion de la tige %s (gui: %i, h5:tige%i)'%(str(tname), cur_tige, tigeid))
    
    # Suppresion de la tige dans le fichier hdf5
    h5store.delete_tige(hdf5file, tigeid)

    # remet a zero les tracés sur la figure
    reset_graph_data()
    
    # recharger les données du fichier hdf5 en mémoire
    load_hdf5_tiges(hdf5file)
    
    plot_image(cur_image, force_clear=True)


###################### Fenetre pour afficher une tige #########################
def export_one_tige():
    """
    Fonction pour exporter les données d'une tige (celle définit par
    la variable globale cur_tige) pour le pas de temps affiché à
    l'écran en csv
    """

    # Id de la tige dans le fichier h5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Le nom de la tige
    tname = h5store.get_tiges_names(hdf5file, hdf_tige_id)
    proposed_filename = "tige_%s_image_%i.csv"%(tname, cur_image+1)
    outfileName = tkFileDialog.asksaveasfilename(parent=toptige,
                                              filetypes=[("Comma Separated Value, csv","*.csv")],
                                              title="Export des données tige %s"%tname,
                                              initialfile=proposed_filename,
                                              initialdir=base_dir_path)
    if len(outfileName) > 0:
        #Creation du tableau avec pandas
        tx = tiges_data.xc[cur_tige,cur_image]
        ty = tiges_data.yc[cur_tige,cur_image]
        tsmoothx, tsmoothy, tangle, ts, tN = traite_tige2(tx, ty,
                                                          tiges_data.diam[cur_tige,cur_image]/2.0,
                                                          pas=PAS_TRAITEMENT)
        tsmoothx = tsmoothx[:-1]
        tsmoothy = tsmoothy[:-1]
        tcourbure = diff(tangle)/diff(ts)
        tcourbure = hstack((tcourbure,tcourbure[-1]))
        data_tmp = {'tige':[tname]*len(tsmoothx),'image':[cur_image+1]*len(tsmoothy),
                    'angle (deg)': tangle,'x (pix)': tsmoothx,'y (pix)': tsmoothy,
                    'abscisse curviligne (pix)': ts, 'courbure c (deg/pix)':tcourbure,
                    'angle_0_360 (pix)': convert_angle(tangle)}

        data_tmp = pd.DataFrame(data_tmp, columns=['tige','image','angle (deg)','x (pix)',
                                                   'y (pix)', 'abscisse curviligne (pix)',
                                                   'courbure c (deg/pix)','angle_0_360 (pix)'])

        data_tmp.to_csv(outfileName, index=False)

        print("Export finit")
        
def save_tige_options():
    """
    Fonction pour sauvegarder les options modifiées dans le gui
    (fonction show_tiges_options) dans le fichier hdf5
    """
    
    reset_graph_data()

    new_tige_name = tktigeid.get()
    new_offset = tk_tige_offset.get()
    id_tige_h5 = get_hdf5_tigeid(hdf5file, cur_tige)
    
    try:
        h5store.save_tige_name(hdf5file, id_tige_h5, new_tige_name)
        # Mise a jour des données de la tiges dans le fichier hdf5
        new_data_tige = {"Tiges_seuil_offset": new_offset}
        #besoin de recup l'id de la tiges dans le fichier hdf5
        h5store.save_tige_dict_to_hdf(hdf5file, id_tige_h5,
                                      new_data_tige)
        
    except Exception as e:
        print(u"La mise à jour des options a raté")
        print(e)


    # Rechargement des données des tiges a partir du hdf5
    load_hdf5_tiges(hdf5file)
    
    #Replot the main image
    plot_image(cur_image, keep_zoom=True,
               force_clear=True)


toptige = None
tk_tige_offset = None
def show_tige_options():
    global toptige, tktigeid, tk_tige_offset

    global floatmenuisopen
    # Force la fermeture du menu popup dans tk
    floatmenuisopen = False
    
    # Récupération des données du fichier hdf5 et de la variable tige_id_mapper
    tname = tiges_names[cur_tige]
    id_tige_hdf = get_hdf5_tigeid(hdf5file, cur_tige)
    seuil = h5store.get_postprocessing(hdf5file, 'Tiges_seuil_offset',
                                       id_tige_hdf)

    # Ajout d'une boite tk pour l'export
    toptige = Tk.Toplevel(master=root)
    toptige.title("Réglage des option pour la tige: tige%i du fichier hdf"%id_tige_hdf)

    #Case pour changer nom de la tige
    idframe = Tk.Frame(toptige)
    Tk.Label(idframe, text='Nom de la tige:').pack(fill='x', expand=True)
    tktigeid = Tk.Entry(idframe)
    tktigeid.insert(0, str(tname))
    tktigeid.pack(fill='x', expand=True)

    Tk.Label(idframe, text='Sensibilité du seuil\n (offset n x seuil):').pack(fill='x', expand=True)
    tk_tige_offset = Tk.DoubleVar()

    if seuil is not None:
        tk_tige_offset.set(seuil)

    #label = Tk.Label(idframe, textvariable=tk_tige_offset).pack(fill='x', expand=True)
    w2 = Tk.Scale(idframe, from_=-5, to=5, resolution=0.1, variable=tk_tige_offset, orient=Tk.HORIZONTAL)
    w2.pack(fill='x', expand=True)

    Tk.Button(idframe,text="Sauvegarder", command = save_tige_options).pack(fill='x', expand=True)

    idframe.pack(fill='x', expand=True)

    tigebutton_export = Tk.Button(master=toptige,
                                  text='Exporter la tige a t=%i vers (csv)'%(cur_image+1),
                                  command=export_one_tige)
    
    tigebutton_export.pack(side=Tk.BOTTOM)


Lgz = 0 # global variable for the length of the growth zone, L_gz
def show_growth_length():
    """Estimation interactive de la longueur de la zone de croissance, L_gz.
    """
    global floatmenuisopen
    # Force la fermeture du menu popup dans tk
    floatmenuisopen = False

    # Recup l'id de la tige dans le fichier h5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Les données enregistrées dans L_data
    L_data = h5store.get_postprocessing(hdf5file, 'L_data', hdf_tige_id)
    # L'échelle si elle a été définie
    scale_cmpix = h5store.get_pixelscale(hdf5file)
    
    if L_data is not None and 'num_img_fit' in L_data:
        cur_tps = L_data['num_img_fit'] #Temps (ou photo) sur laquel on prend la forme finale
    else:
        cur_tps = -1

    def Curve_zone(ds_start, Arad, cur_tps):
        s = cumsum(sqrt(sdx[cur_tps]**2 + sdy[cur_tps]**2))

        if scale_cmpix is not None:
            s *= scale_cmpix

        Atest = Arad[cur_tps]

        Sc = s[ds_start:] - s[ds_start]

        AA0 = Arad[cur_tps, ds_start:] / Arad[cur_tps, ds_start]
        #Prise en compte des données non masquées
        Sc = Sc[~AA0.mask]
        signal = ma.log(AA0[:len(Sc)])

        return Sc, AA0, signal

    def fit_As(Sc, signal, cur_tps):
        global Lgz

        min_func = lambda p, x, y: sum(sqrt((x*p[0] - y)**2))
        min_func_exp = lambda p, x, y: sum(sqrt((ma.exp(- x / p[0]) - y)**2))
        #p0 = signal.std()/Sc[~signal.mask].std()
        #print(p0)

        opt_params = fmin(min_func, [1.], args=(Sc[~signal.mask], signal[~signal.mask]))
        opt_params_exp = fmin(min_func_exp, [1.],
                              args=(Sc[~signal.mask], ma.exp(signal[~signal.mask])))
        #fitA = poly1d( ma.polyfit(Sc, log(AA0), 1) )
        #print(opt_params)
        #print(opt_params_exp)
        #print(fitA)

        Lc = - 1 / opt_params[0]
        Lgz = Sc.max()
        Si = sinit - sinit[0]
        Lo = Si.max()
        Lgz = min((Lgz, Lo))
        if cur_tps == -1:
            print_tps = dx.shape[0]
        else:
            print_tps = cur_tps

        length_scale='pix'
        if scale_cmpix != None:
            length_scale = 'cm'

        text_infos.set_text("Img: %i, unit: %s, Lgz=%0.2f, Ltot=%0.2f"%(
                            print_tps, length_scale, Lgz, Lo))

    def OnPick(evt):
        #Connect dragable lines
        dlines.on_press(evt)

    def OnMotion(evt):
        dlines.on_motion(evt)

    def On_close(evt):
        global L_data

        #Save the data to tige_id_map.pkl
        is_start = find(sinit >= pl_seuil_tps.get_data()[0][0])[0]
        icur_img = int(pl_curv_img.get_data()[1][0])

        #Check if it is possible
        if icur_img >= dx.shape[0]-1:
            icur_img = dx.shape[0]-1

        unit='pix'
        if scale_cmpix != None:
            unit = 'cm'

        L_data[cur_tige] = {'s0': is_start, 'num_img_fit': icur_img, 'Lgz': Lgz,
                            'unit': unit}

        save_tige_idmapper()
        print(L_data[cur_tige])

    def OnRelease(evt):

        dlines.on_release(evt)

        if dlines.changed:
            dlines.changed = False
            #Update final position of both lines
            #pl_seuil_tps.set_xdata([evt.xdata]*2)
            #pl_seuil_tps2.set_xdata([evt.xdata]*2)

            cur_tps = int(pl_curv_img.get_data()[1][0])
            if cur_tps >= dx.shape[0]:
                cur_tps = -1

            ds_start = find(sinit >= pl_seuil_tps.get_data()[0][0])[0]
            pl_seuil_pts.set_data(tiges_data.xc[cur_tige, 0, ds_start] - imgxmi,
                                  tiges_data.yc[cur_tige, 0, ds_start] - imgymi)

            Sc, AA0, signal = Curve_zone(ds_start, A, cur_tps)
            fit_As(Sc, signal, cur_tps)

            scur = cumsum(sqrt(sdx[cur_tps]**2 + sdy[cur_tps]**2))

            if scale_cmpix != None:
                scur *= scale_cmpix

            if cur_tps == -1:
                print_tps = dx.shape[0]-1
            else:
                print_tps = cur_tps

            pl_average.set_data(scur[~A[cur_tps, :].mask],
                                A[cur_tps, ~A[cur_tps, :].mask] - A[cur_tps, 0])
            pl_average.set_label('Img %i'%print_tps)
            ax2.legend(loc=0, prop={'size': 10})
            pl_photo_cur_tige.set_data(tiges_data.xc[cur_tige, cur_tps] - imgxmi,
                                       tiges_data.yc[cur_tige, cur_tps] - imgymi)
            fig.canvas.draw()

    #Need init time tige data
    ti_xc, ti_yc = tiges_data.xc[cur_tige, 0], tiges_data.yc[cur_tige, 0] #raw tige xc,yc

    xmax = max(tiges_data.xc[cur_tige, :].flat)
    xmin = min(tiges_data.xc[cur_tige, :].flat)
    ymax = max(tiges_data.yc[cur_tige, :].flat)
    ymin = min(tiges_data.yc[cur_tige, :].flat)

    imgxma = int(xmax + 0.02*xmax)
    imgxmi = int(xmin - 0.02*xmin)
    imgyma = int(ymax + 0.02*ymax)
    imgymi = int(ymin - 0.02*ymin)

    #print((imgxmi, imgymi, imgxma-imgxmi, imgyma-imgymi))
    fig = mpl.figure('Estimation of the growth length for organ %i'%(cur_tige),
                     figsize=(12, 10))

    G = mpl.GridSpec(2, 4)

    ax1 = mpl.subplot(G[:2, 2:])

    Nimgs = h5store.get_number_of_images(hdf5file)
    image_debut, image_fin = 0, Nimgs - 1

    tmpi = h5store.open_hdf5_image(hdf5file, image_debut, 0)
    imgi = tmpi[imgymi:imgyma, imgxmi:imgxma]
    ax1.imshow(imgi, 'gray')

    tmpf = h5store.open_hdf5_image(hdf5file, image_fin, 0)
    imgf = tmpf[imgymi:imgyma, imgxmi:imgxma]
    ax1.imshow(imgf, 'gray', alpha=0.5)

    ax1.plot(tiges_data.xc[cur_tige, 0] - imgxmi,
             tiges_data.yc[cur_tige, 0] - imgymi, 'g-', lw=2)

    pl_photo_cur_tige, = ax1.plot(tiges_data.xc[cur_tige,cur_tps] - imgxmi,
                                  tiges_data.yc[cur_tige,cur_tps] - imgymi, 'm--', lw=2)
    ax1.axis('equal')
    ax1.axis('off')

    # Calcul du spatio-temporel
    cutoff = 5
    xt, yt = tiges_data.xc[cur_tige, :, :-cutoff], tiges_data.yc[cur_tige, :, :-cutoff]
    lx, ly = xt.shape
    dx, dy = diff(xt, 1), diff(-yt, 1)

    sdx, sdy = zeros_like(dx) - 3000, zeros_like(dy) - 3000
    W = int(round((tiges_data.diam[cur_tige].mean() / 2.) / PAS_TRAITEMENT) * 2.0)

    wind = ones(W, 'd')
    for i in xrange(dx.shape[0]):
        dxT = ma.hstack([dx[i, W-1:0:-1], dx[i,:], dx[i, -1:-W:-1]])
        dyT = ma.hstack([dy[i, W-1:0:-1], dy[i,:], dy[i, -1:-W:-1]])
        cx = convolve(wind / wind.sum(), dxT, mode='valid')[(W/2 - 1):-W/2 - 1]
        cy = convolve(wind / wind.sum(), dyT, mode='valid')[(W/2 - 1):-W/2 - 1]
        sdx[i, :len(cx)] = cx
        sdy[i, :len(cy)] = cy

    sdx = ma.masked_less_equal(sdx, -100.)
    sdy = ma.masked_less_equal(sdy, -100.)

    Arad = ma.arctan2(-sdx, sdy)
    A = rad2deg(Arad)

    sinit = cumsum(sqrt(sdx[0]**2 + sdy[0]**2))
    sfinal = cumsum(sqrt(sdx[-1]**2 + sdy[-1]**2))
    scur = cumsum(sqrt(sdx[cur_tps]**2 + sdy[cur_tps]**2))

    lscale = 'pix'
    if scale_cmpix != None:
        sinit *= scale_cmpix
        scur *= scale_cmpix
        lscale = 'cm'

    ax2 = mpl.subplot(G[0,:2])
    if cur_tps == -1:
        print_tps = dx.shape[0] - 1
    else:
        print_tps = cur_tps

#     ax2.plot(scur[~A[-1, :].mask], A[-1, ~A[-1, :].mask]-A[-1, 0], 'gray', label='Last')
    pl_average, = ax2.plot(scur, A[cur_tps, :] - A[cur_tps, 0],
                           'm', label='Img %i'%print_tps)
    pl_first, = ax2.plot(sinit, A[0, :] - A[0, 0], 'g', label='First')
    seuil = (A[0, :len(A[0][~A[0].mask])/2] - A[0, 0]).mean()
    xlims = ax2.get_xlim()
    pl_seuil_A, = ax2.plot(xlims, [seuil]*2, 'r--')
    mpl.xlabel('s (%s)'%lscale)
    mpl.ylabel(r'Angle $A - A(s=0)$ (deg)')

    # Find intersection with an offset
    if L_data is not None and 's0' in B_data:
        ds_start = B_data['s0']
    else:
        try:
            ds_start = len(A.mean(0)) - find(abs((A.mean(0) - A.mean(0)[0]) - seuil)[::-1] < 5.0)[0]
        except:
            ds_start = 0

    # Check if the value is possible otherwise put the value to the half of s
    if ds_start >= len(sinit[~sinit.mask]):
        ds_start = int(len(sinit[~sinit.mask])/2)

    pl_seuil_tps, = ax2.plot([sinit[ds_start]]*2, ax2.get_ylim(), 'k--', picker=10)
    pl_seuil_pts, = ax1.plot(tiges_data.xc[cur_tige, 0, ds_start] - imgxmi,
                             tiges_data.yc[cur_tige, 0, ds_start] - imgymi, 'ro', ms=12)
    ax2.legend(loc=0, prop={'size': 10})

    ax3 = mpl.subplot(G[1, :2], sharex=ax2)
    print(sfinal, type(sfinal), sfinal.shape)
    print(dx.shape[0])
    colorm = ax3.pcolormesh(sfinal[~sfinal.mask], arange(dx.shape[0]), A[:, ~sfinal.mask])
    ax3.set_ylim(0, dx.shape[0])
    cbar = mpl.colorbar(colorm, use_gridspec=True, pad=0.01)
    cbar.ax.tick_params(labelsize=10)
    pl_seuil_tps2, = ax3.plot([sinit[ds_start]]*2, ax3.get_ylim(), 'k--', picker=10)
    if cur_tps == -1:
        tmpy = dx.shape[0] - 1
    else:
        tmpy = cur_tps

    pl_curv_img, = ax3.plot(ax3.get_xlim(), [tmpy]*2, 'm--', picker=10)
    mpl.ylabel('Num Photo')
    mpl.xlabel('s (%s)'%lscale)

    text_infos = mpl.figtext(.5, 0.01, '', fontsize=11, ha='center', color='Red')

    # Plot the growth zone (i.e. the one that is curved)
    Sc, AA0, signal = Curve_zone(ds_start, A, cur_tps)
    fit_As(Sc, signal, cur_tps)

    dlines = DraggableLines([pl_seuil_tps, pl_seuil_tps2, pl_curv_img],
                            linked=[pl_seuil_tps,pl_seuil_tps2])
    fig.canvas.mpl_connect('pick_event', OnPick)
    fig.canvas.mpl_connect('button_release_event', OnRelease)
    fig.canvas.mpl_connect('motion_notify_event', OnMotion)
    fig.canvas.mpl_connect('close_event', On_close)

    mpl.tight_layout()
    mpl.gcf().show()


tktigeid = None
def show_one_tige(tige_id=None):
    global cur_tige, toptige
    global floatmenuisopen
    # Force la fermeture du menu popup dans tk
    floatmenuisopen = False

    if tige_id is not None:
        #print(tige_id)
        cur_tige = int(tige_id)

    # On choppe l'id de la tige dans le fichier hdf5
    tige_hdf_id = get_hdf5_tigeid(hdf5file, cur_tige)
    # Le nom de la tige
    tname = h5store.get_tiges_names(hdf5file, tige_hdf_id)

    # La mise à l'échelle
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Récuperation des données dont on a besoin
    tige_x, tige_y, tige_taille, tige_angle, tige_measure_zone, tige_lines =\
            load_postprocessed_data(hdf5file, tige_hdf_id)

    # Création de la figure avec le nom de la tige
    figt = mpl.figure('tige %s'%str(tname), figsize=(10, 6))

    G = mpl.GridSpec(4, 3, wspace=.7, hspace=1)

    # Gestion de l'axe du temps (a-t-on des temps ou des numéros d'image)
    dtps = arange(len(tige_taille))
    if get_photo_datetime.get() and dtphoto != []:
        tps = dtphoto
        #Temps en min
        tps = array([(t - tps[0]).total_seconds() for t in tps]) / 60.
        xlabel = 'Temps (min)'
    else:
        tps = dtps
        xlabel = 'N photos'

    #A = convert_angle(tiges_angles[cur_tige])
    tax1 = figt.add_subplot(G[:,0])
    lcollec = LineCollection(tige_lines, linewidth=(2,), color='gray')
    lcollec.set_array(dtps)
    tax1.add_collection(lcollec)
    tax1.set_xlim((tige_x.min(), tige_x.max()))
    tax1.set_ylim((tige_y.min(), tige_y.max()))
    tax1.set_xlabel('x-x0 (pix)')
    tax1.set_ylabel('y-y0 (pix)')
    tax1.axis('equal')

    #Affiche les zones de calcul de l'angle moyen
    #colors = cm.jet(linspace(0,1,len(tps)))
    xt = tige_x[0, ~tige_x[0].mask]
    yt = tige_y[0, ~tige_y[0].mask]
    istart = int(tige_measure_zone[0][0])
    istop = int(tige_measure_zone[0][1])

    try:    
        xlims = [xt[istart], xt[istop]]
        ylims = [yt[istart], yt[istop]]
        colortige, = tax1.plot(xt, yt, 'k', lw=2.5)
        lims, = tax1.plot(xlims, ylims,'o', color='m', ms=10)
    except Exception as e:
        print('Erreur dans le "plot" de la forme de la tige')
        print(e)
        print(istart, istop)

    # Creation des subplot
    tax2 = figt.add_subplot(G[2:, 1:])
    tax3 = figt.add_subplot(G[:2, 1:], sharex=tax2)

    graduation = 60   # une graduation par 60 minutes
    tax2.grid(True)
    tax2.xaxis.set_major_locator(MultipleLocator(graduation))
    tax3.grid(True)
    tax3.xaxis.set_major_locator(MultipleLocator(graduation))

    #Affiche les timeseries Angle au bout et Taille
    if len(tps) > 1:
        if scale_cmpix == None:
            tax2.plot(tps, tige_taille, '+-', color=tiges_colors[cur_tige], lw=2)
            tax2.set_ylabel('Taille (pix)')
        else:
            tax2.plot(tps, tige_taille*scale_cmpix, '+-',
                      color=tiges_colors[cur_tige], lw=2 )
            tax2.set_ylabel('Taille (cm)')

        tax2.set_xlabel(xlabel)

        if angle_0_360.get():
            A = convert_angle(tige_angle)
        else:
            A = tige_angle

        tax3.plot(tps, A, '+-',  color=tiges_colors[cur_tige], lw=2 )
        tax3.set_ylabel('Tip angle (deg)')
        tax3.set_xlabel(xlabel)
 
    else:
        #Si une seul image est traité on montre Angle(f(s)) et Courbure(f(s))
        tsmoothx, tsmoothy, tangle, ts, tN \
                = traite_tige2(tiges_data.xc[cur_tige, 0], tiges_data.yc[cur_tige, 0],
                               tiges_data.diam[cur_tige, 0] /2., pas=PAS_TRAITEMENT)
        #Recentre tout entre 0 et 2pi (le 0 est verticale et rotation anti-horraire)
        tangle = rad2deg(tangle)
        if angle_0_360.get():
            tangle = convert_angle(tangle)

        if scale_cmpix is None:
            tax2.plot(ts, tangle, color=tiges_colors[cur_tige], lw=2)
            tax3.plot(ts[:-5], diff(tangle[:-4]) / diff(ts[:-4]),
                      color=tiges_colors[cur_tige], lw=2)
            tax3.set_xlabel('Abscice curviligne, s (pix)')
            tax3.set_ylabel('Courbure (deg/pix)')
        else:
            tax2.plot(ts*scale_cmpix, tangle, color=tiges_colors[cur_tige], lw=2)
            tax3.plot(ts[:-5]*scale_cmpix, diff(tangle[:-4])/diff(ts[:-4]*scale_cmpix),
                      color=tiges_colors[cur_tige], lw=2)
            tax3.set_xlabel('Abscice curviligne, s (cm)')
            tax3.set_ylabel('Courbure (deg/cm)')

        tax2.set_ylabel('Angle (deg)')

    def on_close(event):
        global toptige
        # Gestion fermeture de la figure
        try:
            toptige.destroy()
            toptige = None
        except:
            pass
        
        mpl.close('tige %s'%str(tname))

    #Add click event for the figure
    figt.canvas.mpl_connect('close_event', on_close)

    #Show the figure
    figt.show()


def show_gamma(tige_id=None, pas=0.3):
    """Estimation interactive du paramètre de proprioception gamma.

    Ce code s'applique aux résultats issus du protocole expérimental suivant :
        1. La plante est initialement à l'horizontal, clinostat éteint.
        2. Elle se redresse progressivement pour se rapprocher de la verticale. Durant
           cette phase, la graviception est dominante.
        3. Le clinostat est mis en route. La plante ne réagit plus qu'à sa courbure et se
           rectifie donc progressivement. C'est la phase de proprioception pure.

    Durant la phase de proprioception pure, la tige relaxe exponentiellement vers une
    forme rectiligne. Si on fait l'hypothèse simplificatrice que la courbure est uniforme
    le long de la tige, alors l'écart entre l'angle au bout et l'angle à la base relaxe
    exponentiellement vers 0, avec le temps caractéristique 1/gamma :
    A_bout - A_base = A0.exp(-gamma*t)

    On estime gamma en ajustant cette exponentielle. La sélection de la plage
    exponentielle (phase de graviception pure) est semi-manuelle. Une plage est inférée
    automatiquement et peut être affinée manuellement à l'aide de taquets mobiles.

    Pour obtenir gamma_tilde, issu du modèle ACĖ (Bastien et al. 2014 Frontiers), il faut
    le taux relatif d'élongation (Relative Elongation Rate, RER). Il est obtenu à partir
    de la courbe de croissance.

    Le code de cette fonction reprend en partie celui de la fonction 'show_one_tige'
    initialement écrit par Hugo Chauvet.
    """
    global cur_tige, toptige

    global floatmenuisopen
    # Force la fermeture du menu popup dans tk
    floatmenuisopen = False
    
    def fit_gamma():
        global gamma_data
        
        xa = select_deb.get_xdata()[0]
        xb = select_fin.get_xdata()[0]
        xdeb = min(xa, xb)
        xfin = max(xa, xb)

        # Indices de début et fin de la plage de fit
        ideb = find(tps >= xdeb)[0]
        ifin = find(tps >= xfin)[0]

        # Tableaux pour le fit de la différence d'angle
        tps_plage = tps[ideb:ifin]
        A_ecart_plage = A_ecart[ideb:ifin]
        log_A_ecart_plage = log_A_ecart[ideb:ifin]

        # La stratégie consiste à passer au log pour se ramener à une régression
        # linéaire. Ce passage pose cependabt un problème du point de vue de la théorie
        # de la régression linéaire. Il y a un biais en faveur des petites valeurs de
        # A_ecart. Le moyen propre d'y remédier est d'ajouter une pondération sous la
        # forme de l'argument 'w=ma.sqrt(A_ecart_plage)' dans l'appel à polyfit.
        # Sources :
        #   - http://mathworld.wolfram.com/LeastSquaresFittingExponential.html
        #   - https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
        weights = ma.sqrt(A_ecart_plage)
        fit_log_A_ecart, res, _, _, _ = ma.polyfit(tps_plage, log_A_ecart_plage, 1,
                                                   w=weights, full=True)

        weighted_mean = mean(weights * log_A_ecart_plage)
        S_res, S_tot = res[0], ma.sum((weights * log_A_ecart_plage - weighted_mean)**2)
        R2 = 1 - S_res / S_tot

        log_A_ecart_fitted = fit_log_A_ecart[0] * tps + fit_log_A_ecart[1]
        plfit_log_A_ecart.set_data(tps_plage, log_A_ecart_fitted[ideb:ifin])
        plfit_A_ecart.set_data(tps[ideb:], exp(log_A_ecart_fitted[ideb:]))

        gamma = -fit_log_A_ecart[0] * 60  # Unité : h¯¹
        gamma_data = {'ideb':ideb, 'ifin':ifin, 'gamma':gamma, 'R2':R2}
        compute_gamma_tilde()
        
    def compute_gamma_tilde():
        global gamma_data

        # Position des taquets mobiles sur le graphe temps-taille
        xa_taille = select_deb_taille.get_xdata()[0]
        xb_taille = select_fin_taille.get_xdata()[0]
        xdeb_taille = min(xa_taille, xb_taille)
        xfin_taille = max(xa_taille, xb_taille)

        # Indices des extrêmités pour l'interpolation
        if xdeb_taille == tps[0] and xfin_taille == tps[-1]:
            # Interpolation par défaut
            taille_deb, taille_fin = taille_min, taille_max
            ideb_taille, ifin_taille = 0, len(tps) - 1
        else:
            # Extrêmités affinées manuellement
            ideb_taille = find(tps >= xdeb_taille)[0]
            ifin_taille = find(tps >= xfin_taille)[0]
            taille_deb = tige_taille[ideb_taille]
            taille_fin = tige_taille[ifin_taille]

        # Tracé de l'interpolation
        if scale_cmpix is None:
            pl_taille_line.set_data([xdeb_taille, xfin_taille], [taille_deb, taille_fin])
        else:
            pl_taille_line.set_data([xdeb_taille, xfin_taille],
                                    [scale_cmpix * taille_deb, scale_cmpix * taille_fin])

        # On veut un Relative Elongation Rate en h¯¹, d'où la multiplication par 60.
        DLDT = 60 * (taille_fin - taille_deb) / (tps[ifin_taille] - tps[ideb_taille])
        RER = DLDT / taille_deb

        gamma = gamma_data['gamma']
        gamma_tilde = gamma / RER

        gamma_data['gamma_tilde'] = gamma_tilde
        gamma_data['RER'] = RER
        gamma_data['ideb_croissance'] = ideb_taille
        gamma_data['ifin_croissance'] = ifin_taille
 
        text_gamma.set_text(r"""$\gamma = %0.4f h^{-1}$"""
                            """\n"""
                            r"""$\tilde{\gamma} = %0.4f$"""
                            """\n"""
                            r"""$\dot{E} = %0.4f h^{-1}$"""
                            """\n"""
                            r"""$R^2 = %0.4f$"""%(gamma, gamma_tilde, RER,
                                                  gamma_data['R2']))

        fig_gamma.canvas.draw_idle()

        # Sauvegarde du gamma_data dans le fichier hdf5
        print('Sauvegarde de gamma_data')
        h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id,
                                      {'gamma_data': gamma_data})
        
    def init_fit():
        # Récupération de gamma_data dans le fichier hdf5
        tigeid = get_hdf5_tigeid(hdf5file, cur_tige)
        gamma_data = h5store.get_postprocessing(hdf5file, 'gamma_data',
                                                 tigeid)
        if gamma_data is not None:
            xdeb = tps[gamma_data['ideb']]
            xfin = tps[gamma_data['ifin']]
            xdeb_taille = tps[gamma_data['ideb_croissance']]
            xfin_taille = tps[gamma_data['ifin_croissance']]
        else:
            # Placement automatique des bornes pour l'ajustement
            try:
                ideb = A_ecart.argmax() + 2
                if any(A_ecart[ideb:] <= 0):
                    ifin = ideb + nonzero(A_ecart[ideb:] <= 0)[0][0] - 3
                else:
                    ifin = -5
                xdeb, xfin = tps[ideb], tps[ifin]
                xdeb_taille, xfin_taille = tps[0], tps[-1]
            except:
                print("""Échec de l'ajustement automatique.\nSélectionnez les bornes
                         manuellement.""")
                ideb, ifin = 0, -1
        try:
            select_deb.set_xdata((xdeb, xdeb))
            select_fin.set_xdata((xfin, xfin))
            select_deb_taille.set_xdata((xdeb_taille, xdeb_taille))
            select_fin_taille.set_xdata((xfin_taille, xfin_taille))
            fit_gamma()
        except:
            print("Erreur dans l'initiation de l'ajustement.")
            print(xdeb, xfin)

    if tige_id is not None:
        #print(tige_id)
        cur_tige = int(tige_id)

    # Recup de l'id de la tige dans le fichier hdf5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)
    # Le nom de la tige
    tname = h5store.get_tiges_names(hdf5file, hdf_tige_id)
    # Récupération des données du scalecmpix dans le fichier hdf5
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Recup des données utiles pour le traitement
    tige_x, tige_y, tige_taille, tige_angle, tige_measure_zone, tige_lines = load_postprocessed_data(hdf5file,
                                                                                                     hdf_tige_id)
    
    fig_gamma = mpl.figure('Estimation de gamma sur la tige %s'%str(tname), figsize=(10, 8))

    G = mpl.GridSpec(6, 3, wspace=.7, hspace=1)
    dtps = arange(len(tige_taille))
    if get_photo_datetime.get() and dtphoto != []:
        tps = dtphoto
        #Temps en dt (min)
        tps = array([(t-tps[0]).total_seconds() for t in tps])/60.
        xlabel='Temps (min)'
    else:
        tps = dtps
        xlabel = 'N photos'

    # Affiche les profiles de tige
    gaxProfiles = fig_gamma.add_subplot(G[:,0])
    lcollec = LineCollection(tige_lines, linewidth=(2,), color='gray')
    lcollec.set_array(dtps)
    gaxProfiles.add_collection(lcollec)
    gaxProfiles.set_xlim((tige_x.min(), tige_x.max()))
    gaxProfiles.set_ylim((tige_y.min(), tige_y.max()))
    gaxProfiles.set_xlabel('x-x0 (pix)')
    gaxProfiles.set_ylabel('y-y0 (pix)')
    gaxProfiles.axis('equal')

    #Affiche les zones de calcul de l'angle moyen
    #colors = cm.jet(linspace(0,1,len(tps)))
    for t in [0]:
        xt = tige_x[t, ~tige_x[t].mask]
        yt = tige_y[t, ~tige_y[t].mask]
        try:
            xlims = [xt[tige_measure_zone[t][0]],
                     xt[tige_measure_zone[t][1]]]
            ylims = [yt[tige_measure_zone[t][0]],
                     yt[tige_measure_zone[t][1]]]
            colortige, = gaxProfiles.plot(xt, yt, 'k', lw=2.5)
            lims, = gaxProfiles.plot(xlims, ylims, 'o', color='m', ms=10)
        except:
            pass

    gaxTaille = fig_gamma.add_subplot(G[4:, 1:])
    gaxAngle = fig_gamma.add_subplot(G[2:4, 1:], sharex=gaxTaille)
    gaxLogAngle = fig_gamma.add_subplot(G[0:2, 1:], sharex=gaxTaille)

    xlims = (tps[0], tps[-1])
    gaxTaille.set_xlim(xlims)
    gaxAngle.set_xlim(xlims)
    gaxLogAngle.set_xlim(xlims)
    gaxAngle.set_ylim((-30, 100))

    graduation = 60   # une graduation par 60 minutes
    gaxTaille.grid(True)
    gaxTaille.xaxis.set_major_locator(MultipleLocator(graduation))
    gaxAngle.grid(True)
    gaxAngle.xaxis.set_major_locator(MultipleLocator(graduation))
    gaxLogAngle.grid(True)
    gaxLogAngle.xaxis.set_major_locator(MultipleLocator(graduation))

    # Les tailles min et max sont resp. le min/max sur les 5 premiers/derniers points.
    taille_min = min(tige_taille[:5])
    taille_max = max(tige_taille[-5:])

    if len(tps) > 1:
        # Tracé de la taille en fonction du temps
        moyenne = (taille_min + taille_max) / 2
        etendue = 300
        if scale_cmpix == None:
            gaxTaille.plot(tps, tige_taille, '+-',
                           color=tiges_colors[cur_tige], lw=2)
            gaxTaille.set_ylim((moyenne - etendue/2, moyenne + etendue/2))
            gaxTaille.set_ylabel('Taille (pix)')
        else:
            gaxTaille.plot(tps, tige_taille*scale_cmpix, '+-',
                           color=tiges_colors[cur_tige], lw=2)
            moyenne *= scale_cmpix
            etendue *= scale_cmpix
            gaxTaille.set_ylim((moyenne - etendue/2, moyenne + etendue/2))
            gaxTaille.set_ylabel('Taille (cm)')

        pl_taille_line, = gaxTaille.plot([], [], 'b', lw=1.5)
        gaxTaille.set_xlabel(xlabel)

        if angle_0_360.get():
            A = convert_angle(tige_angle)
        else:
            A = tige_angle

        # Il faut soustraire l'angle en début de tige A(s = 0)
        xc, yc = tiges_data.xc[cur_tige], tiges_data.yc[cur_tige]
        dx, dy = diff(xc, 1), diff(-yc, 1)
        A_s = rad2deg(ma.arctan2(-dx, dy))
        A_s_mean = A_s.mean(axis=0)   # moyenne temporelle
        A_s0_mean = ma.mean(A_s_mean[:10])    # moyenne spatiale en début de tige
        A_ecart = A - A_s0_mean

        log_A_ecart = log(A_ecart)
        gaxLogAngle.plot(tps, log_A_ecart, '+-',
                         color=tiges_colors[cur_tige], lw=2)
        gaxLogAngle.set_ylabel(r'$\log(A_{bout} - A_{base})$')
        gaxLogAngle.set_xlabel(xlabel)

        gaxAngle.plot(tps, A_ecart, '+-',
                      color=tiges_colors[cur_tige], lw=2)
        gaxAngle.set_ylabel(r'$A_{bout} - A_{base}$ (deg)')
        gaxAngle.set_xlabel(xlabel)

        # Taquets mobiles pour définir la plage de fit
        try:
            select_deb, = gaxLogAngle.plot([tps[5]] * 2, gaxLogAngle.get_ylim(), 'b',
                                           lw=2, picker=5)
            select_fin, = gaxLogAngle.plot([tps[-5]] * 2, gaxLogAngle.get_ylim(), 'b',
                                           lw=2, picker=5)
        except:
            print('Not enough images for estimation of gamma.')
            select_deb, = gaxLogAngle.plot([tps[0]] * 2, gaxLogAngle.get_ylim(), 'b',
                                           lw=2, picker=5)
            select_fin, = gaxLogAngle.plot([tps[0]] * 2, gaxLogAngle.get_ylim(), 'b',
                                           lw=2, picker=5)

        # Le taux de croissance relatif (Relative Elongation Rate, RER) est calculé par
        # défaut sur toute la durée de l'expérience. Il est possible de définir
        # manuellement une plage plus restreinte avec un second couple de taquets mobiles.
        select_deb_taille, = gaxTaille.plot([tps[0]] * 2, gaxTaille.get_ylim(), 'b',
                                            lw=2, picker=5)
        select_fin_taille, = gaxTaille.plot([tps[-1]] * 2, gaxTaille.get_ylim(), 'b',
                                            lw=2, picker=5)
 
        plfit_A_ecart, = gaxAngle.plot([], [], 'b', lw=1.5)
        plfit_log_A_ecart, = gaxLogAngle.plot([], [], 'b', lw=1.5)
        dlines = DraggableLines([select_deb, select_fin,
                                 select_deb_taille, select_fin_taille])
        xmin, xmax = gaxProfiles.get_xlim()
        text_gamma = gaxProfiles.text(xmin + (xmax - xmin) / 10,
                                      gaxProfiles.get_ylim()[-1], '')
        init_fit()
    else:
        print("Zarbi. Une seule image.")
        #Si une seul image est traité on montre Angle(f(s)) et Courbure(f(s))
        tsmoothx, tsmoothy, tangle, ts, tN = traite_tige2(
                tiges_data.xc[cur_tige, 0],
                tiges_data.yc[cur_tige, 0],
                tiges_data.diam[cur_tige, 0] / 2.,
                pas=PAS_TRAITEMENT)
        #Recentre tout entre 0 et 2pi (le 0 est vertical et rotation anti-horaire)
        #tangle[tangle<0] += 2*pi
        tangle = rad2deg(tangle)
        if angle_0_360.get():
            tangle = convert_angle(tangle)
        if scale_cmpix == None:
            gaxTaille.plot(ts, tangle, color=tiges_colors[cur_tige], lw=2)
            gaxAngle.plot(ts[:-5], diff(tangle[:-4])/diff(ts[:-4]),
                          color=tiges_colors[cur_tige], lw=2)
            gaxAngle.set_xlabel('Abscisse curviligne, s (pix)')
            gaxAngle.set_ylabel('Courbure (deg/pix)')
        else:
            gaxTaille.plot(ts*scale_cmpix, tangle,
                            color=tiges_colors[cur_tige], lw=2 )
            gaxAngle.plot(ts[:-5]*scale_cmpix,
                          diff(tangle[:-4])/diff(ts[:-4]*scale_cmpix),
                          color=tiges_colors[cur_tige], lw=2 )
            gaxAngle.set_xlabel('Abscisse curviligne, s (cm)')
            gaxAngle.set_ylabel('Courbure (deg/cm)')

        gaxTaille.set_ylabel('Angle (deg)')

    def OnPick(e):
        dlines.on_press(e)

    def OnMotion(e):
        dlines.on_motion(e)

    def OnRelease(e):
        dlines.on_release(e)
        if gaxLogAngle.contains(e)[0]:
            # Ajustement pour trouver gamma puis calcul du RER pour obtenir gamma_tilde
            fit_gamma()
        if gaxTaille.contains(e)[0]:
            # Pas besoin de réestimer gamma, seulement le RER pour obtenir gamma_tilde
            compute_gamma_tilde()

    def on_close(event):
        global toptige
        #Gestion fermeture de la figure
        try:
            toptige.destroy()
            toptige = None
        except:
            pass


        mpl.close('tige %s'%str(tname))   # faut-il changer ça ?

    #Add click event for the figure
    fig_gamma.canvas.mpl_connect('close_event', on_close)
    fig_gamma.canvas.mpl_connect('pick_event', OnPick)
    fig_gamma.canvas.mpl_connect('button_release_event', OnRelease)
    fig_gamma.canvas.mpl_connect('motion_notify_event', OnMotion)

    #Show the figure
    fig_gamma.show()


tktigeid = None
def show_beta(tige_id=None):
    global cur_tige, toptige, tktigeid, toptige
    global floatmenuisopen
    # Force la fermeture du menu popup dans tk
    floatmenuisopen = False

    def fit_beta(xdeb, xfin):
        #Fonction pour fitter A(t), et L(t) pour calculer b =1/sin(a(t=0)) * Da/dt/Dl/dt * R
        ideb = find(tps>=xdeb)[0]
        ifin = find(tps>=xfin)[0]

        good_tps = tps[ideb:ifin]
        good_A = A[ideb:ifin]
        good_L = tige_taille[ideb:ifin]

        #Need init time tige data
        ti_xc, ti_yc = tiges_data.xc[cur_tige,0], tiges_data.yc[cur_tige,0] #raw tige xc,yc
        dx, dy = diff(ti_xc,1), diff(-ti_yc,1)
        Atotti = ma.arctan2( -dx, dy )

        fitA = ma.polyfit(good_tps, good_A, 1)
        fitL = ma.polyfit(good_tps, good_L, 1)

        xfit = linspace(xdeb, xfin)
        plfitA.set_data(xfit, fitA[0]*xfit+fitA[1])

        if scale_cmpix is None:
            plfitL.set_data(xfit, fitL[0]*xfit+fitL[1] )
        else:
            plfitL.set_data(xfit, scale_cmpix*(fitL[0]*xfit+fitL[1]) )

        figt.canvas.draw_idle()

        DADT = abs(deg2rad(fitA[0]))
        DLDT = fitL[0]
        Ainit = abs(Atotti.mean()) #Angle moyen de la tige au temps 0 (deja en radiant)

        R = tiges_data.diam[cur_tige].mean()/2.
        Runit = R
        DLDTunit = DLDT
        lscale = 'pix'
        if scale_cmpix == None:
            printR = r'$R=%0.2f$ (pix)'%R
        else:
            printR = r'$R=%0.2f$ (cm)'%(R*scale_cmpix)
            Runit *= scale_cmpix
            DLDTunit *= scale_cmpix
            lscale = 'cm'

        Betat = R/sin(Ainit) * (DADT/DLDT)
        print("""R=%0.2f (%s), sin(Ainit)=%0.2f, DADT=%0.3f (rad/min),"""
              """DLDT=%0.3f (%s/min), DADT/DLDT=%0.3f, Betatilde=%0.4f"""%(
                  Runit, lscale, sin(Ainit), DADT, DLDTunit, lscale, DADT/DLDT, Betat))
        textB.set_text(r"""$\tilde{\beta} = %0.4f$"""
                       """\n"""
                       r"""%s"""%(Betat, printR))

        # Sauvegarde des données dans le fichier hdf5

        # Attention pas de / dans les clefs car sinon ça créer un sous
        # groupe dans le fichier hdf5 car '/toto/tata/tutu' veut dire
        # groupe toto puis groupe tata et datatset tutu
        datadict = {"Beta_data":{'ideb':ideb, 'ifin':ifin, 'R':Runit,
                                 'Ainit(rad)':Ainit,'DADT(rad.min-1)':DADT,
                                 'DLDT(lenght unit.min-1)':DLDTunit,
                                 'Betatilde':Betat}}

        try:
            print('Sauvegarde de Beta')
            h5store.save_tige_dict_to_hdf(hdf5file, tige_hdf_id, datadict)
        except Exception as e:
            print('Erreur dans la sauvegarde de Beta:')
            print(e)

        #textB.set_x(xfin)

    def init_fit():
        # Récupération des données dans le fichier hdf5
        Beta_data = h5store.get_postprocessing(hdf5file, "Beta_data",
                                               tige_hdf_id)
        
        if Beta_data is not None:
            xdeb = tps[Beta_data['ideb']]
            xfin = tps[Beta_data['ifin']]
            try:
                fit_beta(xdeb, xfin)
                select_deb.set_xdata((xdeb,xdeb))
                select_fin.set_xdata((xfin,xfin))
            except:
                print('Error in init_fit')
                print(xdeb, xfin)
                xdeb = tps[5]
                xfin = tps[-5]


    if tige_id is not None:
        #print(tige_id)
        cur_tige = int(tige_id)

    # On choppe l'id de la tige dans le fichier hdf5
    tige_hdf_id = get_hdf5_tigeid(hdf5file, cur_tige)
    # Le nom de la tige
    tname = h5store.get_tiges_names(hdf5file, tige_hdf_id)

    # La mise à l'échelle
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Récuperation des données dont on a besoin
    tige_x, tige_y, tige_taille, tige_angle, tige_measure_zone, tige_lines = load_postprocessed_data(hdf5file,
                                                                                                     tige_hdf_id)
    # Création de la figure avec le nom de la tige
    figt = mpl.figure('tige %s'%str(tname), figsize=(10,6))

    G = mpl.GridSpec(4, 3, wspace=.7, hspace=1)

    # Gestion de l'axe du temps (a-t-on des temps ou des numéros d'image)
    dtps = arange(len(tige_taille))
    if get_photo_datetime.get() and dtphoto != []:
        tps = dtphoto
        #Temps en dt (min)
        tps = array([(t-tps[0]).total_seconds() for t in tps])/60.
        xlabel='dt (min)'
    else:
        tps = dtps
        xlabel = 'N photos'

    #A = convert_angle(tiges_angles[cur_tige])
    tax1 = figt.add_subplot(G[:,0])
    lcollec = LineCollection(tige_lines, linewidth=(2,), color='gray')
    lcollec.set_array(dtps)
    tax1.add_collection(lcollec)
    tax1.set_xlim((tige_x.min(), tige_x.max()))
    tax1.set_ylim((tige_y.min(), tige_y.max()))
    tax1.set_xlabel('x-x0 (pix)')
    tax1.set_ylabel('y-y0 (pix)')
    tax1.axis('equal')

    #Affiche les zones de calcul de l'angle moyen
    #colors = cm.jet(linspace(0,1,len(tps)))
    xt = tige_x[0, ~tige_x[0].mask]
    yt = tige_y[0, ~tige_y[0].mask]
    istart = int(tige_measure_zone[0][0])
    istop = int(tige_measure_zone[0][1])

    try:
        xlims = [xt[istart], xt[istop]]
        ylims = [yt[istart], yt[istop]]
        colortige, = tax1.plot(xt, yt, 'k', lw=2.5)
        lims, = tax1.plot(xlims, ylims,'o', color='m', ms=10)
    except Exception as e:
        print('Erreur dans le "plot" de la forme de la tige')
        print(e)
        print(istart, istop)


    # Creation des subplot
    tax2 = figt.add_subplot(G[3:,1:])
    tax3 = figt.add_subplot(G[1:3,1:], sharex=tax2)
    tax4 = figt.add_subplot(G[0,1:], sharex=tax2)
    #Affiche les timeseries Angle au bout et Taille
    if len(tps) > 1:
        #tax2 = figt.add_subplot(G[2:,1:])
        if scale_cmpix == None:
            tax2.plot(tps, tige_taille, '+-',
                      color=tiges_colors[cur_tige], lw=2 )
            tax2.set_ylabel('Taille (pix)')
        else:
            tax2.plot(tps, tige_taille*scale_cmpix, '+-',
                      color=tiges_colors[cur_tige], lw=2 )
            tax2.set_ylabel('Taille (cm)')

        plfitL, = tax2.plot([],[],'m', lw=1.5)
        tax2.set_xlabel(xlabel)
        l1, = tax2.plot([tps[0],tps[0]],tax2.get_ylim(),'k--',lw=1.5)

        #tax3 = figt.add_subplot(G[:2,1:],sharex=tax2)
        if angle_0_360.get():
            A = convert_angle(tige_angle)
        else:
            A = tige_angle

        #A = tiges_angles[tige_id]
        tax3.plot(tps, A, '+-',  color=tiges_colors[cur_tige], lw=2 )
        tax3.set_ylabel('Tip angle (deg)')
        tax3.set_xlabel(xlabel)
        l2, = tax3.plot([tps[0], tps[0]], tax3.get_ylim(), 'k--', lw=1.5)
        try:
            select_deb, = tax4.plot([tps[5]]*2, [0,1],'m', lw=3, picker=5)
            select_fin, = tax4.plot([tps[-5]]*2, [0,1],'m', lw=3, picker=5)
        except:
            print('Not enouth images for estimation of beta')
            select_deb, = tax4.plot([tps[0]]*2, [0,1],'m', lw=3, picker=5)
            select_fin, = tax4.plot([tps[0]]*2, [0,1],'m', lw=3, picker=5)

        plfitA, = tax3.plot([],[], 'm', lw=1.5)
        dlines = DraggableLines([select_deb, select_fin])
        #tax4.set_xticks([])
        tax4.set_yticks([])
        textB = tax4.text(tps[-2], 0.5, '', va='center', ha='right')
        init_fit()
    else:
        #Si une seul image est traité on montre Angle(f(s)) et Courbure(f(s))
        tsmoothx, tsmoothy, tangle, ts, tN = traite_tige2(tiges_data.xc[cur_tige,0], tiges_data.yc[cur_tige,0], 
                                                          tiges_data.diam[cur_tige,0]/2.0, pas=PAS_TRAITEMENT)
        #Recentre tout entre 0 et 2pi (le 0 est verticale et rotation anti-horraire)
        #tangle[tangle<0] += 2*pi
        tangle = rad2deg(tangle)
        if angle_0_360.get():
            tangle = convert_angle(tangle)

        if scale_cmpix is None:
            tax2.plot(ts, tangle, color=tiges_colors[cur_tige], lw=2 )
            tax3.plot(ts[:-5], diff(tangle[:-4])/diff(ts[:-4]),
                      color=tiges_colors[cur_tige], lw=2)
            tax3.set_xlabel('Abscice curviligne, s (pix)')
            tax3.set_ylabel('Courbure (deg/pix)')
        else:
            tax2.plot(ts*scale_cmpix, tangle, color=tiges_colors[cur_tige], lw=2)
            tax3.plot(ts[:-5]*scale_cmpix,
                      diff(tangle[:-4])/diff(ts[:-4]*scale_cmpix),
                      color=tiges_colors[cur_tige], lw=2)
            tax3.set_xlabel('Abscice curviligne, s (cm)')
            tax3.set_ylabel('Courbure (deg/cm)')

        tax2.set_ylabel('Angle (deg)')
        tax4.axis('off')

    def OnPick(e):
        dlines.on_press(e)

    def OnMotion(e):
        dlines.on_motion(e)

    def OnRelease(e):
        dlines.on_release(e)

        if tax4.contains(e)[0]:
            #Update fillbetween_x
            xa = select_deb.get_xdata()[0]
            xb = select_fin.get_xdata()[0]
            xdeb = min(xa,xb)
            xfin = max(xa,xb)
            #print(xdeb, xfin)
            fit_beta(xdeb, xfin)

    def OnClick(event):

        if event.xdata is not None and (tax3.contains(event)[0] or tax2.contains(event)[0]):

            t = int(round(event.xdata))
            #Min distance to clicked point
            t = ((tps - event.xdata)**2).argmin()
            if len(tps) > 1:
                xt = tige_x[t, ~tige_x[t].mask]
                yt = tige_y[t, ~tige_y[t].mask]
                try:
                    xlims = [xt[int(tige_measure_zone[t][0])],
                             xt[int(tige_measure_zone[t][1])]]
                    ylims = [yt[int(tige_measure_zone[t][0])],
                             yt[int(tige_measure_zone[t][1])]]
                    colortige.set_data(xt,yt)
                    lims.set_data(xlims,ylims)
                except Exception as e:
                    print(e)

                l1.set_xdata([tps[t],tps[t]])
                l2.set_xdata([tps[t],tps[t]])

                #redraw figure
                figt.canvas.draw()

                #Change figure on main frame
                plot_image(t, keep_zoom=True)

    def on_close(event):
        global toptige
        # Gestion fermeture de la figure
        try:
            toptige.destroy()
            toptige = None
        except:
            pass

        mpl.close('tige %s'%str(tname))

    #Add click event for the figure
    figt.canvas.mpl_connect('button_press_event', OnClick)
    figt.canvas.mpl_connect('close_event', on_close)
    figt.canvas.mpl_connect('pick_event', OnPick)
    figt.canvas.mpl_connect('button_release_event', OnRelease)
    figt.canvas.mpl_connect('motion_notify_event', OnMotion)

    #Show the figure
    figt.show()


def show_B(tige_id=None):
    global floatmenuisopen
    # Force la fermeture du menu popup dans tk
    floatmenuisopen = False

    Nimgs = h5store.get_number_of_images(hdf5file)
    get_growth_length(tiges_data, cur_tige, image_debut=0, image_fin=Nimgs-1)
    mpl.gcf().show()


    #mpl.close('Growth lenght')

def plot_moyenne():
    """
    Trace l'ensemble des courbes angles/tailles vs temps pour toutes
    les tiges ainsi que la courbe moyenne
    """
    

    if len(base_tiges) > 1:

        # Creation du tableau Pandas pour stocquer les données
        data_out = pd.DataFrame(columns=['tige', 'angle', 'temps',
                                         'taille', 'rayon', 'angle_0_360', 'sequence'])

        # Récupérations des id des tiges dans le fichier h5
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)
        tiges_names = h5store.get_tiges_names(hdf5file)
        pictures_names = h5store.get_images_names(hdf5file)
        scale_cmpix = h5store.get_pixelscale(hdf5file)
        
        # Récupère le temps
        if get_photo_datetime.get() and dtphoto != []:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))
            
        # Boucle sur les tiges
        data_tmp = [None] * len(base_tiges)
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_taille, tige_angle, tige_zone, _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tname = tiges_names[i]
                
            data_tmp[i] = pd.DataFrame({'tige': tname, 'angle': tige_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'angle_0_360': convert_angle(tige_angle),
                                     'sequence': 0})

        data_out = pd.concat(data_tmp, ignore_index=True)


        # La figure
        legendtige=["tige %s"%str(i) for i in tiges_names]
        fig = mpl.figure(u'Série temporelles avec moyenne',figsize=(10,5))
        G = mpl.GridSpec(2,4)
        ax1 = mpl.subplot(G[0,:3])
        if angle_0_360.get():
            dataa = "angle_0_360"
        else:
            dataa = "angle"


        plot_sequence(data_out, tige_color=tiges_colors, show_lims=False,
                      ydata=dataa, tige_alpha=1.0)
        if not get_photo_datetime.get():
            mpl.xlabel("Number of images")
        mpl.grid()

        mpl.subplot(G[1,:3], sharex=ax1)
        yplot = 'taille'
        if scale_cmpix is not None:
            data_out['taille(cm)'] = data_out['taille'] * scale_cmpix
            yplot = 'taille(cm)'
            
        plot_sequence(data_out, tige_color=tiges_colors, show_lims=False,
                      ydata=yplot, tige_alpha=1)
        ylims = mpl.ylim()
        mpl.ylim(ylims[0]-20, ylims[1]+20)
        if not get_photo_datetime.get():
            mpl.xlabel("Number of images")
        mpl.grid()

        #axl = subplot(G[:,3])
        #print legendtige
        ax1.legend(legendtige+["Moyenne"], bbox_to_anchor=(1.02, 1), loc=2)

        fig.tight_layout()
        fig.show()

    else:
        print(u"Pas de serie temporelle, une seule photo traitée")


def No_output_print(**kwargs):
    pass

def save_tige_idmapper():
    """
    Fonction pour sauvegarder les données de post-processing sur les
    estimation des perceptions et autres

    TODO: A supprimer quand le passage a hdf5 est fini
    """
    print('Save data...')
    
    #Save the tige_id_mapper
    with open(base_dir_path+'tige_id_map.pkl','wb') as f:
        pkl.dump({'tige_id_mapper':tige_id_mapper,
                  'scale_cmpix':scale_cmpix,
                  'L_data': L_data,
                  'G_data': G_data,
                  'B_data': B_data,
                  'Beta_data': Beta_data,
                  'gamma_data': gamma_data,
                  'Tiges_seuil_offset': Tiges_seuil_offset}, f)

#Calcul de la longueur de croissance
GoodL = 0 #Some global variables
Lc = 0
Lcexp = 0
B = 0
Bexp = 0
def get_growth_length(tiges, cur_tige, thresold='auto',
                      image_debut=0, image_fin=None, pas = 0.3):
    """
    Compute the curvature length as described in the AC model of Bastien et al.

    tiges: is the tiges instance
    thresold[auto]: the thresold if computed as 2 times the mean diameter.
    image_debut, donne la position de la première image a afficher dans le fichier h5
    image_fin, donne la position de la dernière image a afficher dans le fichier h5
    """

    # Recup l'id de la tige dans le fichier h5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Les données enregistrées dans B_data
    B_data = h5store.get_postprocessing(hdf5file, 'B_data', hdf_tige_id)
    # L'échelle si elle a été définie
    scale_cmpix = h5store.get_pixelscale(hdf5file)
    
    if B_data is not None and 'num_img_fit' in B_data:
        cur_tps = B_data['num_img_fit'] #Temps (ou photo) sur laquel on prend la forme finale
    else:
        cur_tps = -1

    def compute_R(model, data):
        #Cf wikipedia Coefficient_of_determination
        sstot = ma.sum( (data - data.mean())**2 )
        #ssreg = ma.sum( (model - data.mean())**2 )
        ssres = ma.sum( (model - data)**2 )

        R = 1 - ssres/sstot
        return R

    def Curve_zone(ds_start, Arad, cur_tps):
        s = cumsum(sqrt(sdx[cur_tps]**2 + sdy[cur_tps]**2))
        
        if scale_cmpix is not None:
            s *= scale_cmpix

        Atest = Arad[cur_tps]
        Sc = s[ds_start:] - s[ds_start]
        AA0 = Arad[cur_tps, ds_start:] / Arad[cur_tps, ds_start]
        #Prise en compte des données non masquées
        Sc = Sc[~AA0.mask]
        signal = ma.log(AA0[:len(Sc)])
        pl_A_exp.set_data(Sc, AA0[:len(Sc)])
        pl_A_log.set_data(Sc, signal)

        try:
            ax4.set_xlim(0, Sc.max())
            ax4.set_ylim(AA0.min(), AA0.max())
            ax5.set_xlim(0, Sc.max())
            ax5.set_ylim(signal.min(), signal.max())
        except:
            pass

        return Sc, AA0, signal


    def fit_As(Sc, signal, cur_tps):
        global GoodL, Lc, Lcexp, B, Bexp
        
        min_func = lambda p, x, y: sum( sqrt( (x*p[0] - y)**2 ) )
        min_func_exp = lambda p, x, y: sum( sqrt( (ma.exp(-x/p[0]) - y)**2 ) )
        #p0 = signal.std()/Sc[~signal.mask].std()
        #print(p0)

        opt_params = fmin(min_func, [1.0], args = (Sc[~signal.mask], signal[~signal.mask]))
        opt_params_exp = fmin(min_func_exp, [1.0], args = (Sc[~signal.mask], ma.exp(signal[~signal.mask])))
        #fitA = poly1d( ma.polyfit(Sc, log(AA0), 1) )
        #print(opt_params)
        #print(opt_params_exp)
        #print(fitA)
        Lc = -1/opt_params[0]
        Lgz = Sc.max()
        Si = sinit-sinit[0]
        Lo = Si.max()
        GoodL = min((Lgz,Lo))
        B = GoodL/Lc
        Lcexp = opt_params_exp[0]
        Bexp = GoodL/Lcexp
        if cur_tps == -1:
            print_tps = dx.shape[0]
        else:
            print_tps = cur_tps

        length_scale='pix'
        if scale_cmpix is not None:
            length_scale = 'cm'

        text_infos.set_text("Img: %i, unit: %s, Lzc=%0.2f, Ltot=%0.2f || fit (A/A0): Lc=%0.2f, B=%0.2f || fit log(A/A0): Lc=%0.2f, B=%0.2f"%(print_tps, length_scale, Lgz, Lo, Lcexp, Bexp, Lc, B))

        xtmp = linspace(0,max(mpl.gca().get_xlim()))
        fit4log.set_data(xtmp, exp(-xtmp/Lc))
        fit4exp.set_data(xtmp, exp(-xtmp/opt_params_exp[0]))

        Rlogexp = compute_R(ma.exp(-Sc[~signal.mask]/Lc), ma.exp(signal[~signal.mask]))
        Rexp = compute_R(ma.exp(-Sc[~signal.mask]/opt_params_exp[0]), ma.exp(signal[~signal.mask]))
        fit4log.set_label(r'$R^2 = %0.3f$'%Rlogexp)
        fit4exp.set_label(r'$R^2 = %0.3f$'%Rexp)
        ax4.legend(loc=0,prop={'size':10})

        xtmp = linspace(0,max(mpl.gca().get_xlim()))
        fit5.set_data(xtmp, -xtmp/Lc)

        Rlog = compute_R(-Sc[~signal.mask]/Lc, signal[~signal.mask])
        fit5.set_label(r'$R^2 = %0.3f$'%Rlog)
        ax5.legend(loc=0,prop={'size':10})


    def OnPick(evt):
        #Connect dragable lines
        dlines.on_press(evt)

    def OnMotion(evt):
        dlines.on_motion(evt)

    def On_close(evt):
        #Do things when close the B windows.

        #Save the data to tige_id_map.pkl
        is_start = find(sinit >= pl_seuil_tps.get_data()[0][0])[0]
        icur_img = int(pl_curv_img.get_data()[1][0])

        #Check if it is possible
        if icur_img >= dx.shape[0]-1:
            icur_img = dx.shape[0]-1

        unit='pix'
        if scale_cmpix is not None:
            unit = 'cm'

        
        newb_data = {'s0': is_start, 'num_img_fit': icur_img, 'Lgz': GoodL,
                     'Lc (fit Log(A over A0))':Lc, 'Lc (fit exp(A over A0))': Lcexp,
                     'unit': unit, 'B (log)': B, 'B (exp)': Bexp}

        # Sauvegarde dans le fichier h5
        datadict = {'B_data': newb_data}
        h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id, datadict)

        print(newb_data)


    def OnRelease(evt):


        dlines.on_release(evt)

        if dlines.changed:
            dlines.changed = False
            #Update final position of both lines
            #pl_seuil_tps.set_xdata([evt.xdata]*2)
            #pl_seuil_tps2.set_xdata([evt.xdata]*2)

            cur_tps = int(pl_curv_img.get_data()[1][0])
            if cur_tps >= dx.shape[0]:
                cur_tps = -1

            ds_start = find(sinit >= pl_seuil_tps.get_data()[0][0])[0]
            pl_seuil_pts.set_data(tiges.xc[cur_tige,0,ds_start]-imgxmi,
                                  tiges.yc[cur_tige,0,ds_start]-imgymi)

            Sc, AA0, signal = Curve_zone(ds_start, A, cur_tps)
            fit_As(Sc, signal, cur_tps)
            scur = cumsum(sqrt( sdx[cur_tps]**2 + sdy[cur_tps]**2 ))

            if scale_cmpix is not None:
                scur *= scale_cmpix

            if cur_tps == -1:
                print_tps = dx.shape[0]-1
            else:
                print_tps = cur_tps

            pl_average.set_data(scur[~A[cur_tps,:].mask], A[cur_tps,~A[cur_tps,:].mask]-A[cur_tps,0])
            pl_average.set_label('Img %i'%print_tps)
            ax2.legend(loc=0,prop={'size':10})
            pl_photo_cur_tige.set_data(tiges.xc[cur_tige,cur_tps]-imgxmi,
                                       tiges.yc[cur_tige,cur_tps]-imgymi)
            fig.canvas.draw()

    #print(cur_tige)

    #Need init time tige data
    ti_xc, ti_yc = tiges.xc[cur_tige,0], tiges.yc[cur_tige,0] #raw tige xc,yc
    #print(diff(ti_s)[::20])

    #Need last time tige data
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
    fig = mpl.figure('B for organ %i'%(cur_tige), figsize=(12,10))


    G = mpl.GridSpec(3,4)
    ax1 = mpl.subplot(G[:2,2:])

    
    tmpi = h5store.open_hdf5_image(hdf5file, image_debut, 0)
    imgi = tmpi[imgymi:imgyma, imgxmi:imgxma]
    ax1.imshow(imgi, 'gray')

    if image_fin is not None:
        tmpf = h5store.open_hdf5_image(hdf5file, image_fin, 0)
        imgf = tmpf[imgymi:imgyma, imgxmi:imgxma]
        ax1.imshow(imgf, 'gray', alpha=0.5)

    ax1.plot(tiges.xc[cur_tige,0]-imgxmi, tiges.yc[cur_tige,0]-imgymi,
             'g-', lw=2)
    pl_photo_cur_tige, = ax1.plot(tiges.xc[cur_tige,cur_tps]-imgxmi,
                                  tiges.yc[cur_tige,cur_tps]-imgymi, 'm--', lw=2)
    ax1.axis('equal')
    ax1.axis('off')

    #Caclul du spatio-temporel
    coupe_end = 5
    xt, yt = tiges.xc[cur_tige,:,:-coupe_end], tiges.yc[cur_tige,:,:-coupe_end]
    lx, ly = xt.shape
    dx, dy = diff(xt,1), diff(-yt,1)

    sdx, sdy = zeros_like(dx) - 3000, zeros_like(dy) - 3000
    W = int(round( (tiges.diam[cur_tige].mean()/2.)/pas ) * 2.0)

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

    
    sinit = cumsum( sqrt( sdx[0]**2 + sdy[0]**2 ) )
    sfinal = cumsum( sqrt( sdx[-1]**2 + sdy[-1]**2 ) )
    scur = cumsum( sqrt( sdx[cur_tps]**2 + sdy[cur_tps]**2 ) )
    
    lscale = 'pix'
    if scale_cmpix is not None:
        sinit *= scale_cmpix
        scur *= scale_cmpix
        sfinal *= scale_cmpix
        lscale = 'cm'

    ax2 = mpl.subplot(G[0,:2])
    if cur_tps == -1:
        print_tps = dx.shape[0] - 1
    else:
        print_tps = cur_tps

    ax2.plot(scur[~A[-1,:].mask], A[-1,~A[-1,:].mask]-A[-1,0], 'gray', label='Last')
    pl_average, = ax2.plot(scur, A[cur_tps,:]-A[cur_tps,0], 'm', label='Img %i'%print_tps)
    pl_first, = ax2.plot(sinit, A[0,:]-A[0,0], 'g', label='First')
    seuil = (A[0,:len(A[0][~A[0].mask])/2]-A[0,0]).mean()
    xlims = ax2.get_xlim()
    pl_seuil_A, = ax2.plot(xlims,[seuil]*2, 'r--')
    mpl.xlabel('s (%s)'%lscale)
    mpl.ylabel('Angle A-A(s=0) (deg)')

    #Find intersection with an offset
    if B_data is not None and 's0' in B_data:
        ds_start = B_data['s0']
    else:
        try:
            ds_start = len(A.mean(0)) - find( abs( (A.mean(0)-A.mean(0)[0]) - seuil )[::-1] < 5.0 )[0]
        except:
            ds_start = 0

    #check if the value is possible otherwise put the value to the half of s
    if ds_start >= len(sinit[~sinit.mask]):
        ds_start = int(len(sinit[~sinit.mask])/2)



    pl_seuil_tps, = ax2.plot([sinit[ds_start]]*2, ax2.get_ylim(), 'k--', picker=5)
    pl_seuil_pts, = ax1.plot(tiges.xc[cur_tige,0,ds_start]-imgxmi, tiges.yc[cur_tige,0,ds_start]-imgymi, 'ro', ms=12)
    ax2.legend(loc=0,prop={'size':10})
    
    ax3 = mpl.subplot(G[1,:2], sharex=ax2)
    #print(sfinal, type(sfinal), sfinal.shape)
    #print(dx.shape[0])
    colorm = ax3.pcolormesh(sfinal[~sfinal.mask], arange(dx.shape[0]), A[:,~sfinal.mask])
    ax3.set_ylim(0, dx.shape[0])
    cbar = mpl.colorbar(colorm, use_gridspec=True, pad=0.01)
    cbar.ax.tick_params(labelsize=10)
    pl_seuil_tps2, = ax3.plot([sinit[ds_start]]*2, ax3.get_ylim(), 'k--', picker=5)
    if cur_tps == -1:
        tmpy = dx.shape[0] - 1
    else:
        tmpy = cur_tps

    pl_curv_img, = ax3.plot(ax3.get_xlim(), [tmpy]*2, 'm--', picker=5)
    mpl.ylabel('Num Photo')
    mpl.xlabel('s (%s)'%lscale)

    #Fit sur le log(A/A0)
    ax4 = mpl.subplot(G[2,:2])
    pl_A_exp,= ax4.plot([], [], 'o')
    fit4exp, = ax4.plot([], [], 'r', lw=2, label='R')
    fit4log, = ax4.plot([], [], 'g--', lw=2, label='R')
    mpl.ylabel('A/A0')
    mpl.xlabel('Sc (%s)'%lscale)

    ax5 = mpl.subplot(G[2,2:])
    pl_A_log, = ax5.plot([], [], 'o')
    fit5, = ax5.plot([], [], 'g', lw=2, label='R')
    mpl.ylabel('log(A/A0)')
    mpl.xlabel('Sc (%s)'%lscale)

    text_infos = mpl.figtext(.5,0.01,'', fontsize=11, ha='center', color='Red')
    #Plot the growth zone (i.e. the one that curve)
    Sc, AA0, signal = Curve_zone(ds_start, A, cur_tps)
    fit_As(Sc, signal, cur_tps)


    dlines = DraggableLines([pl_seuil_tps,pl_seuil_tps2, pl_curv_img],
                            linked=[pl_seuil_tps,pl_seuil_tps2])
    fig.canvas.mpl_connect('pick_event', OnPick)
    fig.canvas.mpl_connect('button_release_event', OnRelease)
    fig.canvas.mpl_connect('motion_notify_event', OnMotion)
    fig.canvas.mpl_connect('close_event', On_close)

    mpl.tight_layout()


def measure_pixels():
    global add_dist_draw, dist_measure_pts, plt_measure
    #Function to measure the pixel by drawing two points
    if not add_dist_draw:
        add_dist_draw = True
        dist_measure_pts = [] #empty points list to store them
        try:
            plt_measure.set_data([],[])
            canvas.draw()
        except:
            pass


def update_scale_pxcm():
    global pixel_distance, cm_distance

    done = False
    try:
        px = float(pixel_distance.get())
        cm = float(cm_distance.get())
        print("scale %0.4f cm/pix"%(cm/px))
        scale_cmpix = cm/px
        done = True
    except:
        print('Error in scale')
        scale_cmpix = None #reset scale to none

    if done:
        pixel_distance.delete(0, Tk.END)
        cm_distance.delete(0, Tk.END)
        pixel_distance.insert(0, '1.0')
        cm_distance.insert(0, '%0.4f'%scale_cmpix)

        #On sauvegarde dans le fichier h5
        h5store.save_pixelscale(hdf5file, scale_cmpix)

        #On efface le trait sur la figure
        try:
            plt_measure.set_data([],[])
            canvas.draw()
        except:
            pass

def pixel_calibration():
    #Function to calibrate pixel->cm
    global pixel_distance, cm_distance

    def update_cm_distance(sv):
        #new_val = sv.get()
        update_scale_pxcm()

    #Ajout d'une boite tk pour l'export
    topcal = Tk.Toplevel(master=root)
    topcal.title("Calibration pix->cm")

    #Case pour changer nom de la tige
    calframe = Frame(topcal)
    pixel_distance = Entry(calframe, width=10)

    #Récupère l'échelle dans le fichier h5
    scale_cmpix = h5store.get_pixelscale(hdf5file)
          
    if scale_cmpix is not None:
        pixel_distance.insert(0, str('1.0'))

    cm_distance = Entry(calframe, width=10)
    if scale_cmpix is not None:
        cm_distance.insert(0, str('%0.4f'%scale_cmpix))

    Tk.Label(calframe, text='pixel:').pack()
    pixel_distance.pack()
    Tk.Label(calframe, text='cm:').pack()
    cm_distance.pack()
    calframe.pack()

    calbutton_calibration = Button(master=topcal, text='Measure distance',
                                   command=measure_pixels)
    calbutton_calibration.pack(fill=Tk.X)

    calbutton_updatecalibration = Button(master=topcal, text='Update scale',
                                         command=update_scale_pxcm)
    calbutton_updatecalibration.pack(fill=Tk.X)


def trouve_dict_subkeys(input_dict):
    """"
    Fonction pour trouver les clefs d'un dictionnaire contenu dans un autre dictionnaire.
    
    data = {"0": {"a":0,"b":1},
            "1": {"a": 10, "b": 12}}

    La fonction retourne ["a", "b"] ou [] si les sous dictionnaires sont vides
    """

    dict_keys = []
    for key in list(input_dict):
        if input_dict[key] is not None:
            dict_keys = input_dict[key].keys()
            break

    return dict_keys

def _export_tige_id_mapper_to_csv():
    """
    Fonction pour exporter les données de phénotypage des tiges/racines 
    Beta et Gamma etc...
    """

    proposed_filename = "Phenotype_gravi_proprio.csv"
    outfileName = tkFileDialog.asksaveasfilename(parent=root,
                                                  filetypes=[("Comma Separated Value, csv","*.csv")],
                                                  title=u"Export des données de phénotypage",
                                                  initialfile=proposed_filename,
                                                  initialdir=base_dir_path)

    if len(outfileName) > 0:

        # On récupère les données dans le fichier hdf5
        L_data = h5store.get_postprocessing(hdf5file, 'L_data')
        G_data = h5store.get_postprocessing(hdf5file, 'G_data')
        B_data = h5store.get_postprocessing(hdf5file, 'B_data')
        Beta_data = h5store.get_postprocessing(hdf5file, 'Beta_data')
        gamma_data = h5store.get_postprocessing(hdf5file, 'gamma_data')
        tiges_names = h5store.get_tiges_names(hdf5file)
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)

        # Il faut trouver les clés des données standards pour chaque dico
        L_keys = trouve_dict_subkeys(L_data)
        G_keys = trouve_dict_subkeys(G_data)
        B_keys = trouve_dict_subkeys(B_data)
        Beta_keys = trouve_dict_subkeys(Beta_data)
        gamma_keys = trouve_dict_subkeys(gamma_data)

        scale_cmpix = h5store.get_pixelscale(hdf5file)

        ntiges = len(base_tiges)
        output = pd.DataFrame()
        try:
            for i in range(ntiges):
                tmp_out = {}
                tmp_out['scale (cm/pix)'] = scale_cmpix
                tmp_out['id_tige'] = hdf_tiges_id[i]
                tmp_out['tige_name'] = tiges_names[i]

                hdf_id = hdf_tiges_id[i]

                if hdf_id in L_data:
                    for key in L_keys:
                        if L_data[hdf_id] is None:
                            tmp_out["L_%s" % key] = None
                        else:
                            tmp_out["L_%s" % key] = L_data[hdf_id][key]

                if hdf_id in G_data:
                    for key in G_keys:
                        if G_data[hdf_id] is None:
                            tmp_out["G_%s" % key] = None
                        else:
                            tmp_out["G_%s" % key] = G_data[hdf_id][key]

                if hdf_id in B_data:
                    for key in B_keys:
                        if B_data[hdf_id] is None:
                            tmp_out["B_%s" % key] = None
                        else:
                            tmp_out["B_%s" % key] = B_data[hdf_id][key]

                if hdf_id in Beta_data:
                    for key in Beta_keys:
                        if Beta_data[hdf_id] is None:
                            tmp_out["Beta_%s" % key] = None
                        else:
                            tmp_out["Beta_%s" % key] = Beta_data[hdf_id][key]

                if hdf_id in gamma_data:
                    for key in gamma_keys:
                        if gamma_data[hdf_id] is None:
                            tmp_out["gamma_%s" % key] = None
                        else:
                            tmp_out["gamma_%s" % key] = gamma_data[hdf_id][key]

                output = output.append(tmp_out, ignore_index=True)
        except Exception as e:
                print(u"Error dans l'export des données\nError (tige %i): %s"%(i, e))

        print(u"Saved to %s"%outfileName)
        output.to_csv(outfileName, index=False)

###############################################################################

################################ Main windows #################################
if __name__ == '__main__':

    root = Tk.Tk()
    root.style = Style()
    root.style.theme_use("clam")
    root.wm_title("RootStemExtractor -- version:%s" % (__version__))
    print("RootStemExtractor -- version:%s" % (__version__))
    
    #TOP MENU BAR
    menubar = Tk.Menu(root)

    # Export menu
    exportmenu = Tk.Menu(menubar, tearoff=0)
    exportmenu.add_command(label=u"Série temporelle moyenne", command=_export_mean_to_csv)
    exportmenu.add_command(label=u"Séries temporelles par tiges", command=_export_meandata_for_each_tiges)
    exportmenu.add_command(label=u"Séries temporelles globales [A(t,s=tip), L(t)]", command=_export_to_csv)
    exportmenu.add_command(label=u"Séries temporelles globales + squelette", command=_export_xytemps_to_csv)
    exportmenu.add_command(label=u"Phenotype (graviception, proprioception)", command=_export_tige_id_mapper_to_csv)
    menubar.add_cascade(label="Exporter", menu=exportmenu)

    #Plot menu
    plotmenu = Tk.Menu(menubar, tearoff=0)
    plotmenu.add_command(label=u"Série temporelles", command=plot_moyenne)
    menubar.add_cascade(label="Figures", menu=plotmenu)

    #Options menu
    options_menu = Tk.Menu(menubar)
    #Pour chercher le temps dans les données EXIFS des images
    get_photo_datetime = Tk.BooleanVar()
    get_photo_datetime.set(True)
    options_menu.add_checkbutton(label="Extract photo time", onvalue=True, offvalue=False, variable=get_photo_datetime)
    angle_0_360 = Tk.BooleanVar()
    angle_0_360.set(False)
    options_menu.add_checkbutton(label="Angle modulo 360 (0->360)", onvalue=True, offvalue=False, variable=angle_0_360)
    #check_black = Tk.BooleanVar()
    #check_black.set(False)
    #options_menu.add_checkbutton(label="Photos noires (<5Mo)", onvalue=True, offvalue=False, variable=check_black)
    # options_menu.add_command(label=u"Test detection", command=test_detection) # TODO: BUG WITH THREAD
    options_menu.add_command(label=u"Calibration (pix/cm)", command=pixel_calibration)
    #TODO: Pour trier ou non les photos
    #sort_photo_num = Tk.BooleanVar()
    #sort_photo_num.set(True)
    #options_menu.add_checkbutton(label="Sort pictures", onvalue=True, offvalue=False, variable=sort_photo_num)

    menubar.add_cascade(label='Options', menu=options_menu)

    #Display the menu
    root.config(menu=menubar)
    root.columnconfigure(1, weight=1, minsize=600)
    root.rowconfigure(1, weight=1)
    #Floating menu (pour afficher des infos pour une tige)
    floatmenu = Tk.Menu(root, tearoff=0)
    #floatmenu.add_command(label="Inverser la base", command=_reverse_tige)
    floatmenu.add_command(label="Réglages", command=show_tige_options)
    floatmenu.add_command(label="Séries temporelles", command=show_one_tige)
#    floatmenu.add_command(label="Obtenir la longueur de croissance",
#                          command=show_growth_length)
    floatmenu.add_command(label="Obtenir gamma", command=show_gamma)
    floatmenu.add_command(label="Obtenir beta", command=show_beta)
    floatmenu.add_command(label="Obtenir B", command=show_B)
    floatmenu.add_command(label="Suprimer la base", command=remove_tige)
    floatmenuisopen = False


    def popup(tige_id):
        global floatmenuisopen, cur_tige
         # display the popup menu
        cur_tige = tige_id

        try:
            floatmenu.tk_popup(int(root.winfo_pointerx()), int(root.winfo_pointery()))
            floatmenuisopen = True
        finally:
            # make sure to release the grab (Tk 8.0a1 only)
            floatmenu.grab_release()
            #pass


    #BOTTOM MENU BAR
    buttonFrame = Frame(master=root)
    #buttonFrame.pack(side=Tk.BOTTOM)
    buttonFrame.grid(row=0, column=0, sticky=Tk.W)
    button_traiter = Button(master=buttonFrame, text='Traiter', command=launch_process, state=Tk.DISABLED)
    button_listimages = Button(master=buttonFrame, text="Liste d'images", command=show_image_list, state=Tk.DISABLED)
    button_addtige = Button(master=buttonFrame, text='Ajouter une base', command=_addtige, state=Tk.DISABLED)
    button_supr_all_tige = Button(master=buttonFrame, text='Supprimer les bases', command=_supr_all_tiges, state=Tk.DISABLED)
    button_ouvrir = Button(master=buttonFrame, text='Ouvrir', command=_open_files)
    prog_bar = Progressbar(master=root, mode='determinate')
    #Ajout d'un bouton export to csv
    #button_export = Tk.Button(master=buttonFrame, text=u'Exporter série vers (csv)', command=_export_to_csv, state=Tk.DISABLED)

    """
    button_ouvrir.pack(side=Tk.LEFT)
    button_listimages.pack(side=Tk.LEFT)
    button_addtige.pack(side=Tk.LEFT)
    button_supr_all_tige.pack(side=Tk.LEFT)
    button_traiter.pack(side=Tk.LEFT)
    prog_bar.pack(side=Tk.LEFT, padx=10)
    """
    button_ouvrir.grid(row=0, column=0)
    button_listimages.grid(row=0, column=1)
    button_addtige.grid(row=0, column=2)
    button_supr_all_tige.grid(row=0, column=3)
    button_traiter.grid(row=0, column=4)
    prog_bar.grid(row=2, columnspan=2, sticky=Tk.E+Tk.W)
    
    #button_export.pack(side=Tk.LEFT)
    #figsize=(10,8)
    fig = mpl.figure()
    ax = fig.add_subplot(111)
    
    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.show()
    #canvas.get_tk_widget().pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
    canvas.get_tk_widget().grid(row=1, columnspan=2, sticky=Tk.W+Tk.E+Tk.N+Tk.S)
    tbar_frame = Tk.Frame(root)
    tbar_frame.grid(row=0, column=1, sticky="ew")
    toolbar = NavigationToolbar2TkAgg(canvas, tbar_frame)
    toolbar.update()
    #canvas._tkcanvas.pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)

    #On affiche la première image
    plot_image(cur_image)
    canvas._tkcanvas.config(cursor='cross')
    
    def on_key_event(event):
        print(u'you pressed %s'%event.key)
        key_press_handler(event, canvas, toolbar)

        if event.key == '+':
            global add_tige, nbclick
            add_tige = True
            nbclick = 0

        if event.key == 'right':
            if cur_image + 1 < len(h5store.get_images_names(hdf5file)):
                plot_image(cur_image + 1, keep_zoom=True)

        if event.key == 'left':
            if cur_image > 1:
                plot_image(cur_image - 1, keep_zoom=True)

        if event.key == 'escape':
            #Cancel add_tige
            if add_tige:
                add_tige = False
                if nbclick == 1:
                    nbclick = 0
                    base_tiges.pop(-1)

                plot_image(cur_image, keep_zoom=True)
                change_button_state()

    def onClick(event):
        global base_tiges, nbclick, add_tige, btige_plt, btige_text
        global floatmenuisopen
        global plt_measure, add_dist_draw, dist_measure_pts, pixel_distance
        #print event

        #Restore focus on the current canvas
        canvas.get_tk_widget().focus_force()

        if event.button == 1:
            #Manage how to add a tige
            if add_tige:
                xy = (event.xdata,event.ydata)
                if xy[0] != None and xy[1] != None:
                    # On récupère la résolution de l'image affiché
                    scale = used_scale
                    xyscale = [c*scale for c in xy]
                    if nbclick == 0:
                        base_tiges.append([xyscale])
                        nbclick += 1
                        ax.plot(xy[0],xy[1],'r+')
                        canvas.draw()
                    else:
                        base_tiges[-1].append(xyscale)
                        nbclick = 0
                        add_tige = False

                        # Creation de la tige dans le fichier hdf5 On
                        # cree un id qui est plus grand que l'id max
                        # qui se trouve deja dans le fichier hdf5
                        tiges_ids = h5store.get_tiges_indices(hdf5file)
                        print('ids dans le fichier hdf5' + str(tiges_ids))
                        if tiges_ids == []:
                            # c'est la première base
                            new_tige_id = 0
                        else:
                            # On ajoute 1 au max
                            new_tige_id = max(tiges_ids) + 1

                        print('nouvel id pour la tige dans hdf' + str(new_tige_id))
                        h5store.create_tige_in_hdf(hdf5file, new_tige_id)

                        # Besoin de sauvegarder les points de base de la tige
                        h5store.save_base_tige(hdf5file, new_tige_id,
                                               base_tiges[-1])
                        
                        # Besoin de crée un nom pour la tige (ancien
                        # role du tige_id_mapper). On lui donne le nom
                        # qui correspond a son tige_id
                        h5store.save_tige_name(hdf5file,
                                               new_tige_id,
                                               str(new_tige_id))

                        # On efface la figure
                        reset_graph_data()
                        
                        # On recharge les données des tiges dans le GUI
                        load_hdf5_tiges(hdf5file)

                        # Recharge l'image complètement
                        plot_image(cur_image, force_clear=True)
                        
                    change_button_state()

            if add_dist_draw:
                if dist_measure_pts == []:
                    plt_measure, = ax.plot(event.xdata,event.ydata,'yo-', label='measure', zorder=10)
                    dist_measure_pts += [event.xdata,event.ydata]
                    canvas.draw_idle()
                else:
                    plt_measure.set_data( (dist_measure_pts[0], event.xdata), (dist_measure_pts[1],event.ydata) )
                    canvas.draw_idle()
                    if pixel_distance != None:
                        tmpd = sqrt( (dist_measure_pts[0]-event.xdata)**2 + (dist_measure_pts[1]-event.ydata)**2 )
                        pixel_distance.delete(0, Tk.END)
                        pixel_distance.insert(0, str('%0.2f'%tmpd))

                    dist_measure_pts = []
                    add_dist_draw = False

            #Close floatmenu
            if floatmenuisopen:
                floatmenu.unpost()
                floatmenuisopen = False

        if event.button == 3:
            #Cancel add_tige
            if add_tige:
                add_tige = False
                if nbclick == 1:
                    nbclick = 0
                    base_tiges.pop(-1)

                plot_image(cur_image)
                change_button_state()


    def onPick(event):
        
        if isinstance(event.artist, mpl.Line2D):
            #print event.mouseevent
            thisline = event.artist
            #xdata = thisline.get_xdata()
            #ydata = thisline.get_ydata()
            tige_id = thisline.get_label()
            
            # Besoin de savoir si c'est la base ou la tige qui a été pické
            # Pour éviter les ouverture à répétition de la même fenetre

            if 'base_' in tige_id:
                isbase = True
                tige_id = tige_id.replace('base_','')
            else:
                isbase = False

            tige_id_hdf = get_hdf5_tigeid(hdf5file, int(tige_id))
            
            #print('onpick1 line:', zip(take(xdata, ind), take(ydata, ind)))
            try:
                print(u'Selection de la tige %s' % tiges_names[int(tige_id)])
                print(u'Enregistrée dans le fichier hdf sous tige%i ' % tige_id_hdf)
                print(u'Position dans la liste du gui %i' % int(tige_id))
            except:
                print(u'Selection de la tige %i du gui'%(int(tige_id)))
                print(u'Enregistrée sous le nom tige%i dans le fichier hdf' % tige_id_hdf)

            if event.mouseevent.button == 1:
                if not isbase:
                    print('run show tige')
                    show_one_tige(tige_id=int(tige_id))
                
            if event.mouseevent.button == 3:
                if not floatmenuisopen:
                    print('open float menu')
                    popup(int(tige_id))


    def onmousemove(evt):
        """
        Fonction déclancher à chaque mouvement de la souris dans le
        canvas matplotlib
        """
        # print(toolbar._active)
        if toolbar._active == 'PAN':
            canvas._tkcanvas.config(cursor='hand2')

        if toolbar._active == 'ZOOM':
            canvas._tkcanvas.config(cursor='sizing')

        if toolbar._active is None:
            if add_tige:
                canvas._tkcanvas.config(cursor='crosshair')
            else:
                canvas._tkcanvas.config(cursor='arrow')
        #print(cursors.POINTER)
        
    # Connxion de fonctions retours au canvas matplotlib
    cidkey = canvas.mpl_connect('key_press_event', on_key_event)
    canvas.mpl_connect('button_press_event', onClick)
    canvas.mpl_connect('pick_event', onPick)
    canvas.mpl_connect("motion_notify_event", onmousemove)
    
    def onclose():
        root.destroy()
        mpl.close('all')
        #sys.exit(0)

    root.protocol('WM_DELETE_WINDOW', onclose)


    root.mainloop()

