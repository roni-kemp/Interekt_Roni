#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:14:54 2015

Petite interface graphique pour traiter les tiges ou les racines

@authors: Hugo Chauvet and Félix Hartmann

Changes:
09/04/2021: [Félix] Multilingual support for English, Esperanto and French + bufixes.
05/10/2020: [Félix] A lot of changes: bugfixes for Py2/3 compatibility
                    + multilingual support with gettext (English-only for now)
                    + partial refactoring, for the multilingual support
                    + new ways of estimating the growth length
                    + new estimation of B, with the selection of a steady-state image
04/06/2020: [Félix] Transition to Python3, while maintaining compatibility with Python2.
29/05/2020: [Félix] Adding growth rate estimation for computing beta_tilde and beta_tilde
                    + improved export of phenotyping data to CSV file.
20/01/2020: [Félix] Adding Lagrangian marks
                    + popup window for manualy estimating the growth length using curvature heatmap
06/09/2019: [Hugo] Fin de la première version compatible HDF5
19/07/2019: [Félix] Ajout du calcul de gamma pour expérience sous clinostat, Hugo ajout de la fonction reset_globals_data
04/05/2019: [Hugo] Correct bugs on export when tige_id_mapper was defined with string names for bases. Allow float in the slide to select sensitivity.
24/09/2018: [Hugo] Remove test dectection (ne marche pas en Thread avec matplotlib!) + Correction bug pour traiter une seule image.
08/06/2018: [Hugo] Correct queue.qsize() bug in osX (car marche pas sur cette plateforme
18/04/2018: [Hugo] correct datetime bug. Set percent_diam to 1.4 in MethodOlivier (previous 0.9). Windows system can now use thread
22/10/2017: [Hugo] Optimisation du positionnement du GUI, utilisation de Tk.grid a la place de Tk.pack
16/10/2015: [Hugo] Ajout de divers options pour la tige (suppression etc..) avec menu click droit
                   +Refactorisation de plot_image et de la gestion du global_tige_id_mapper (pour gerer les suppressions)

30/07/2015: [Hugo] Correction bugs pour les racines dans libgravimacro + Ajout visu position de la moyenne de l'ange + Ajout d'options pour le temps des photos
25/05/2015: [Hugo] Ajout des menus pour l'export des moyennes par tiges et pour toutes les tiges + figures
20/05/2015: [Hugo] Première version
"""

from __future__ import (unicode_literals, absolute_import, division, print_function)

import sys, os
if sys.version_info[0] < 3:
    python2 = True
else:
    python2 = False

if python2:
    import Tkinter as Tk, tkFileDialog as filedialog, tkMessageBox as messagebox
    from ttk import Style, Button, Frame, Progressbar, Label, Entry, Scale, Checkbutton
    import Queue as queue
else:
    import tkinter as Tk
    from tkinter import filedialog, messagebox
    from tkinter.ttk import (Style, Button, Frame, Progressbar, Label, Entry, Scale,
                             Checkbutton)
    import queue

if python2:
    import cPickle as pkl

import matplotlib
matplotlib.use('TkAgg')

#Remove zoom key shorcuts
matplotlib.rcParams['keymap.back'] = 'c'
matplotlib.rcParams['keymap.forward'] = 'v'

import matplotlib.pylab as mpl
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import SymLogNorm
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import key_press_handler
if python2:
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2TkAgg as NavigationToolbar2Tk)
else:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from numpy import (array, arange, zeros_like, zeros, ones, linspace, newaxis, NaN, isnan,
                   hstack, ma, exp, log, arctan2, sqrt, sin, cos, pi, mean,
                   timedelta64, argmin, argsort, nonzero, flatnonzero, diff, gradient,
                   rad2deg, deg2rad, quantile,
                   )
from numpy.linalg import norm

from scipy.optimize import fmin
import pandas as pd

import re
finddigits = re.compile(r'\d+?')
from threading import Thread
from collections import OrderedDict

import gettext  # internationalization

from new_libgravimacro import (ProcessImages, traite_tiges2, traite_tige2,
                               get_differential_arc_length,
                               integrate_diff_arc_length,
                               get_tige_curvature,
                               convert_angle,
                               plot_sequence,
                               do_intersect,
                               load_results) # 'load_results' is old method for pkl files

import interekt_hdf5_store as h5store

from Extrawidgets import DraggableLines, DraggablePoints

__version__ = '2020-10-05'

########################## GLOBAL DATA ########################################
strings = dict()  # strings for the user interface, with gettext
tiges_data = None
tiges_names = []
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
add_mark = False
marks = []
mark_plt = []
mark_text = []
dtphoto = []
thread_process = None  # To store the thread that run image processing
infos_traitement = None
old_step = 0
add_dist_draw = False
dist_measure_pts = []
pixel_distance = None
cm_distance = None
interekt = None # contains the main window
hdf5file = None # path to hdf5 data file
tk_list_images = None # Tk listbox object which contain the list of images
tk_toplevel_listimages = None # Tk window which displays the list of images
tk_tige_offset = None # contains the offset for tige skeleton detection
tk_tige_percent_diam = None # contains the diameter multplier for the transverse range of tige skeleton detection
tk_detection_step = None  # contains the space step for the skeleton detection process
PERCENT_DIAM_DEFAULT = 1.4    # default value chosen by Hugo
DETECTION_STEP_DEFAULT = 0.3  # default value chosen by Hugo
CURVATURE_AVERAGING_LENGTH = 2  # Length (in cm) of the zone over which the curvature is averaged

def reset_graph_data():
    global img_object, text_object, tiges_plot_object
    global btige_plt, btige_text, btige_arrow
    global mark_plt, mark_text

    img_object = None
    text_object = None
    tiges_plot_object = []
    btige_plt = []
    btige_text = []
    btige_arrow = []
    mark_plt = []
    mark_text = []


def reset_globals_data():
    """
    Reset all global variables, ala Fortran.

    Useful when loading a new file.
    """
    print("Reset all variables and data.")

    global data_out, text_object, tige_plot_object, base_pts_out, base_dir_path
    global cur_image, cur_tige, toptige, add_tige, btige_arrow
    global nbclick, base_tiges, btige_plt, btige_text, tiges_colors
    global marks, mark_plt, mark_text
    global dtphoto, local_photo_path, thread_process, infos_traitement
    global old_step, add_dist_draw, dist_measure_pts, pixel_distance, cm_distance
    global tk_list_images, tk_toplevel_listimages
    global tk_tige_offset, tk_tige_percent_diam, tk_detection_step
    global hdf5file, tiges_data, tiges_names

    # Reset all gl;obalvariables
    # RAZ des autres valeurs
    data_out = None
    img_object = None
    text_object = None
    tige_plot_object = None
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
    add_mark = False
    marks = []
    mark_plt = []
    mark_text = []
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
    tk_tige_offset = None
    tk_tige_percent_diam = None
    tk_detection_step = None
    hdf5file = None
    tiges_data = None
    tiges_names = []

    # On ferme la liste des images si elle est ouverte
    if tk_toplevel_listimages:
        tk_toplevel_listimages.destroy()
        tk_toplevel_listimages = None

###############################################################################

########################## CONVERT OLD PKL TO HDF5 ############################

def convert_pkl_to_hdf(pklfile, hdf5file, display_message=False, display_progress=False):
    """Converts the olf format .pkl into hdf5."""
    #TODO localize strings
    global tiges_data

    base_dir_path = os.path.dirname(pklfile) + '/'

    # On charge le fichier pkl
    data_out = load_results(pklfile)

    # On convertit les images (avec deux images réduites)

    if display_message:
        interekt.display_message(
                """Ancien format de données: CONVERSION (rootstem_data.pkl -> """
                """interekt_data.h5)\nConversion des images...""")

    Nimgs = len(data_out['tiges_info'])
    for tps_step in data_out['tiges_info']:
        # Test si les images sont disponibles à partir du chemin
        # d'origine ou dans le même dossier que le fichier
        # RootStemData.pkl
        if os.path.exists(tps_step['imgname']):
            img_path = tps_step['imgname']
        else:
            img_path = base_dir_path + os.path.basename(tps_step['imgname'])

        # On affiche l'état d'avancement
        if display_message:
            interekt.display_message(
                    """Ancien format de données: CONVERSION (rootstem_data.pkl -> """
                    """interekt_data.h5)\nConversion des images..."""
                    """(%i/%i)"""%(tps_step['iimg'], Nimgs))

        if display_progress:
            # On augmente la barre de progression du GUI
            dstep = 1/float(Nimgs) * 100.
            #print(dstep)
            interekt.prog_bar.step(dstep)
            interekt.prog_bar.update_idletasks()
            interekt.master.update()

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

    if display_progress:
        # Remise a zero de la progress bar
        interekt.prog_bar.step(0.0)
        interekt.prog_bar.update_idletasks()
        interekt.master.update()

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
            for dname in ('scale_cmpix', 'growth_data', 'B_data', 'beta_data',
                          'gamma_data', 'End_point_data', 'Tiges_seuil_offset'):

                if dname in datapkl:
                    exec("%s = datapkl['%s']" % (dname, dname))
                else:
                    #Creation d'un dico vide
                    exec("%s = {}" % (dname))
    else:
        tige_id_mapper = {}

    # Création des dossiers des tiges et du tige_id_mapper si il n'a
    # pas été defini dans le fichier tige_id_map.pkl
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

        if display_message:
            interekt.display_message(
                    """Ancien format de données: CONVERSION (rootstem_data.pkl -> """
                    """interekt_data.h5)\nConversion des squelettes..."""
                    """(tige %i/%i)"""%(id_tige+1, len(base_tiges)))

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
        for dname in ('growth_data', 'B_data', 'beta_data', 'gamma_data',
                      'End_point_data', 'Tiges_seuil_offset'):
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

###############################################################################

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
        print(e, "Update Matplotlib!")
        tiges_colors = mpl.cm.hsv(linspace(0, 1, s))

def get_hdf5_tigeid(h5file, gui_tige_id):
    """Returns the id under which the tige is saved in the hdf5 file."""

    tiges_ids = h5store.get_tiges_indices(h5file)
    tigeid = tiges_ids[gui_tige_id]

    return tigeid

def load_hdf5_tiges(hdf5file, display_message=True):
    """
    Fonction pour recharger toutes les tiges depuis le fichier hdf5
    vers les variables globales d'Interekt.

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
    if display_message:
        interekt.display_message(_("""Loading data"""))
    tiges_names = h5store.get_tiges_names(hdf5file)
    base_tiges = h5store.get_tiges_bases(hdf5file)
    #Recharge la colormap pour les tiges ouvertes
    set_tiges_colormap()

    #Traitement des données
    print("Loading data from the h5 file")
    tiges_data = h5store.get_tigesmanager(hdf5file)

    # For each tige, let's check whether useful data have been stored by 'Traite_data'
    for tige_id in range(Ntiges):
        # Let's get the id of the tige in the hdf5 file
        hdf_tige_id = get_hdf5_tigeid(hdf5file, tige_id)
        angle = h5store.get_postprocessing(hdf5file, 'angle', hdf_tige_id)
        if angle is None:
            Traite_data(save_to_hdf5=True)

def load_hdf5_file(hdf5file):
    """Loads a hdf5 file into Interekt."""
    global dtphoto

    # On charge les données des tiges
    load_hdf5_tiges(hdf5file)

    interekt.get_photo_datetime.set(True)

    # Doit on charger le temps des photos
    #if interekt.get_photo_datetime.get():
    dtphoto = h5store.get_images_datetimes(hdf5file)

    # On regarde si tous les temps sont des datetime sinon on
    # prends le numéro de la photo
    for t in dtphoto:
        if t is None:
            print("Error: No timestamp for all photos; photo numbers are used instead.")
            dtphoto = []
            break

    # Si pas de temps dans les photos on mets l'option dans
    # l'interface graphique à False
    if dtphoto == []:
        interekt.get_photo_datetime.set(False)

    # Retrieve the data on the steady state from the h5 file
    steady_state, exclude_steady_state = h5store.get_steady_state(hdf5file)
    interekt.steady_state_image = steady_state
    interekt.exclude_steady_state_from_time_series.set(exclude_steady_state)


######################### HANDLING OF THE IMAGE LIST ##########################

def onChangeImage(event):
    try:
        sender = event.widget
        idx = sender.curselection()
        #print("idx %i"%idx)
        interekt.plot_image(idx[0], keep_zoom=True)
    except Exception as e:
        print("Erreur de chargement !!!")
        print(e)

def show_image_list():
    global tk_list_images, tk_toplevel_listimages

    if tk_toplevel_listimages is None:
        tk_toplevel_listimages = Tk.Toplevel(master=root)
        tk_toplevel_listimages.title(_("List of open images"))

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
    """Safely closed the image list window."""
    global tk_list_images, tk_toplevel_listimages

    # On détruit la fenetre
    tk_toplevel_listimages.destroy()

    # On remet les variables global à None (Fortran Style)
    tk_list_images = None
    tk_toplevel_listimages = None

###############################################################################


########################## HANDLING OF TIMES ##################################

def get_time_data():
    """Returns useful variables for plotting time series.

    Returns:
    --------
    - times: 1D array of photo times
    - time_units: dictionary with time-related units
    - time_label: label for the time axis in plots (str)
    - graduation: on the time axis of a plot, a tick will be placed
      at each multiple of 'graduation'
    - excluded_image: index of the image to be excluded from time
      series (otberwise, 'None')
    - mask: a mask array for excluding the above-mentioned image
    """

    nb_images = h5store.get_number_of_images(hdf5file)

    # We define a mask which will be useful if the steady-state image
    # has to be excluded from time series.
    mask = ones(nb_images, dtype=bool)
    if (interekt.steady_state_image is not None
            and interekt.exclude_steady_state_from_time_series.get()):
        excluded_image = interekt.steady_state_image
        mask[excluded_image] = False
    else:
        excluded_image = None

    if interekt.get_photo_datetime.get() and dtphoto:
        # photos have timestamps

        # time in minutes
        times = array([(t - dtphoto[0]).total_seconds() for t in dtphoto]) / 60.

        # total duration of the experiment in minutes
        total_minutes = times[mask].max()

        # choose time unit and graduation for the graphs,
        # based on the total duration
        if total_minutes < 12 * 60:  # less than 12 hours
            time_unit = strings["min"]
            graduation = 60   # a graduation every hour
        elif total_minutes < 24 * 60:  # between 12 and 24 hours
            times /= 60  # time in hours
            time_unit = strings["hours"]
            graduation = 1   # a graduation every hour
        elif total_minutes <  7 * 24 * 60:  # between one day and one week
            times /= 60  # time in hours
            time_unit = strings["hours"]
            graduation = 24   # a graduation every day
        elif total_minutes < 30 * 24 * 60:  # between one week and one month
            times /= 24 * 60  # time in days
            time_unit = strings["days"]
            graduation = 1   # a graduation every day
        elif total_minutes < 3 * 30 * 24 * 60:  # between one month and three month
            times /= 24 * 60  # time in days
            time_unit = strings["days"]
            graduation = 7   # a graduation every week
        elif total_minutes >= 3 * 30 * 24 * 60:  # more than three month
            times /= 24 * 60  # time in days
            time_unit = strings["days"]
            graduation = 30   # a graduation every 30 days

        time_label = strings["time"] + " (%s)"%time_unit
        time_units = {"time unit": time_unit,
                      "RER unit": "h-1",
                      "RER unit TeX": r"h$^{-1}$",
                      "dAdt unit": "rad.h-1",
                      "dAdt unit TeX": r"rad.h$^{-1}$"}

    else:
        times = arange(1, nb_images+1)  # time = image number

        # one tick every 60 minutes if photos taken every 6 minutes
        # (this is a legacy from the Murinas project)
        graduation = 10

        time_label = strings["image number"]
        RER_unit = strings["image"] + "-1"
        RER_unit_TeX = strings["image"] + r"$^{-1}$"
        time_units = {"time unit": strings["image"],
                      "RER unit": RER_unit,
                      "RER unit TeX": RER_unit_TeX,
                      "dAdt unit": "rad." + RER_unit,
                      "dAdt unit TeX": r"rad." + RER_unit_TeX}

    return times, time_units, time_label, graduation, excluded_image, mask


##########W####################################################################


########################## Pour le Traitement #################################
def _export_to_csv():
    """
    Fonction pour exporter la serie temporelle pour les tiges soit
    l'évolution de l'angle moyen au bout et de la taille au cours du
    temps
    """

    if len(base_tiges) > 0:
        proposed_filename = "time_series_tipangle_length.csv"
        outfileName = filedialog.asksaveasfilename(
                parent=root,
                filetypes=[("Comma Separated Value, csv","*.csv")],
                title=_("Export time series"),
                #title=_("Export séries temporelles"),
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
        if interekt.get_photo_datetime.get() and dtphoto:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))

        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_s, tige_taille, tige_angle, tige_tip_angle, tige_zone,\
                    _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tige_name = tiges_names[i]

            data_tmp = pd.DataFrame({'tige': tige_name, 'angle': tige_tip_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'angle_0_360': convert_angle(tige_tip_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)

        #Add some usefull data
        #Ntige = data_out.tige.unique()
        output = []
        #print Ntige
        for tige, datatige in data_out.groupby('tige'):
            # print(dataout.tige)

            #Compute dt (min)
            if interekt.get_photo_datetime.get():
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
        proposed_filename = "skeleton.csv"
        outfileName = filedialog.asksaveasfilename(
                parent=root,
                filetypes=[("Comma Separated Value, csv","*.csv")],
                title=_("Export skeleton"),
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
        if interekt.get_photo_datetime.get() and dtphoto:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))

        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_s, tige_taille, tige_angle, tige_tip_angle, tige_zone,\
                    _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tige_name = tiges_names[i]

            data_tmp = pd.DataFrame({'tige': tige_name, 'angle': tige_tip_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'x': tige_x.tolist(),
                                     'y': tige_y.tolist(),
                                     'nom photo': pictures_names,
                                     'angle_0_360': convert_angle(tige_tip_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)

        print(data_out.head())
        print("Enregistré dans %s" % outfileName)
        data_out.to_csv(outfileName, index=False)

def _export_mean_to_csv():
    """
    Fonction qui exporte la courbe moyennée pour toutes les tiges de
    l'angle et taille en fonction du temps
    """

    if len(base_tiges) > 0:
        proposed_filename = "Serie_tempotelle_moyenne.csv"
        outfileName = filedialog.asksaveasfilename(
                parent=root,
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
        if interekt.get_photo_datetime.get() and dtphoto:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))

        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_s, tige_taille, tige_angle, tige_tip_angle, tige_zone,\
                    _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tige_name = tiges_names[i]

            data_tmp = pd.DataFrame({'tige': tige_name, 'angle': tige_tip_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'angle_0_360': convert_angle(tige_tip_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)


        #Creation de la moyenne
        datamoy = data_out.groupby('temps').mean()
        datamoy['temps'] = data_out.temps.unique()
        datamoy['tige'] = ['%s->%s'%(str(data_out['tige'].min()),
                                     str(data_out['tige'].max()))]*len(datamoy['temps'])

        #Convert to timedelta in minute
        if interekt.get_photo_datetime.get():
            dtps = (datamoy['temps']-datamoy['temps'][0])/timedelta64(1,'m')
            datamoy['dt (min)'] = dtps

        datamoy['pictures name'] = pictures_names
        print("Saved to %s"%outfileName)
        datamoy.to_csv(outfileName, index=False)

def _export_meandata_for_each_tiges():
    """
    Dans un dossier donnée par l'utilisateur on exporte les données
    taille et angle au cours de temps avec un fichier .csv par
    tiges/racines
    """

    #Ask for a directory where to save all files
    outdir = filedialog.askdirectory(
            title="Choisir un répertoire pour sauvegarder les tiges")
    if len(outdir) > 0:
        # Creation du tableau Pandas pour stocquer les données
        data_out = pd.DataFrame(columns=['tige', 'angle', 'temps',
                                         'taille', 'rayon', 'angle_0_360'])

        # Récupérations des id des tiges dans le fichier h5
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)
        tiges_names = h5store.get_tiges_names(hdf5file)
        pictures_names = h5store.get_images_names(hdf5file)

        # Récupère le temps
        if interekt.get_photo_datetime.get() and dtphoto:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))

        # Boucle sur les tiges
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_s, tige_taille, tige_angle, tige_tip_angle, tige_zone,\
                    _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tige_name = tiges_names[i]

            data_tmp = pd.DataFrame({'tige': tige_name, 'angle': tige_tip_angle,
                                     'temps': tps, 'taille': tige_taille,
                                     'rayon': tige_R.mean(1),
                                     'angle_0_360': convert_angle(tige_tip_angle)})

            data_out = data_out.append(data_tmp, ignore_index=True)

        #Loop over tiges
        # Ntige = dataframe.tige.unique()
        for tige, datatige in data_out.groupby('tige'):
            #Compute dt (min)
            if interekt.get_photo_datetime.get():
                dtps = (datatige['temps']-datatige['temps'][datatige.index[0]])/timedelta64(1,'m')
                datatige['dt (min)'] = dtps

            datatige['pictures name'] = pictures_names
            # Trouve la position de la tige dans les données du GUI
            itige = tiges_names.index(tige)
            datatige['x base (pix)'] = [tiges_data.xc[itige,:,0].mean()]*len(datatige.index)

            #print datatige.head()
            outfileName = outdir+'/data_mean_for_tige_%s.csv'%str(tige)
            print("Sauvegardé dans %s"%outfileName)
            datatige.to_csv(outfileName, index=False)

###############################################################################

def update_tk_progress(infos, root):
    global old_step
    if infos is not None:

        im_num = infos['inum']
        old_num = infos['old_inum']
        tot = infos['tot']
        start_msg = _("Image processing")
        msg = start_msg + " %i / %i"%(int(im_num),int(tot))
        root.wm_title("Interekt | %s"%msg)
        root.update_idletasks()

        #New version with Tk progressbar
        if im_num != old_step and tot > 1:
            old_step = im_num
            #print(old_step, nstep, nstep-old_step)
            dstep = (im_num-old_num)/float(tot-1) * 100.
            #print(dstep)
            interekt.prog_bar.step(dstep)
            interekt.prog_bar.update_idletasks()
            root.update()

def plot_progress(**kwargs):
    global infos_traitement
    infos_traitement = kwargs

def none_print(**kwargs):
    output = kwargs

def check_process():
    global root, infos_traitement

    if thread_process.is_alive():
        update_tk_progress(infos_traitement, root)
        root.after(20, check_process)

    else:
        update_tk_progress(infos_traitement, root)
        root.wm_title("Interekt")
        thread_process.join()
        infos_traitement = None
        try:
            process_data()
        except Exception as e:
            print('Failed to process data')
            print(e)

def process_data(display_message=True):
    global data_out, tiges_data

    # Get queue result
    data_out = data_out.get()

    # When it's done get the tiges_data from the data from data_out
    tiges_data = data_out['data']['tiges_data']

    # On affiche que l'on fait la sauvegarde
    if display_message:
        interekt.display_message(_("Saving data into the .h5 file"))
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
    if display_message:
        interekt.display_message(_("Processing extracted data"))

    Traite_data(save_to_hdf5=True)

    # Recharge les données des tiges a partir du fichier hdf5
    load_hdf5_tiges(hdf5file)

    # Plot the first image in the list
    interekt.plot_image(cur_image, force_clear=True)

    interekt.change_button_state()
    interekt.prog_bar.stop()

    # Restore focus on the current canvas
    interekt.canvas.get_tk_widget().focus_force()

def launch_process():
    """
    Fonction pour lancer le traitement des images chargées dans
    Interekt pour les tiges tracées.
    """
    global data_out, old_step, thread_process, Crops_data


    Crops_data = []
    #Windows OK in thread now
    is_thread = True

    if len(base_tiges) > 0:

        #Create a process to start process image (that can into all processors)
        #is_thread = False
        reset_graph_data()
        data_out = queue.Queue()

        # On regarde le nombre d'image que l'on a
        Nimgs = h5store.get_number_of_images(hdf5file)

        # Retrieve the detection step from the Tkinter DoubleVar
        if tk_detection_step is not None:
            detection_step = tk_detection_step.get()
        else:
            detection_step = DETECTION_STEP_DEFAULT

        # Save the value of detection step into the hdf5 file
        try:
            h5store.save_detection_step(hdf5file, detection_step)
        except Exception as e:
            print("Failure to update the detection step")
            print(e)

        # Récup les options pour les tiges (seuil de coupure et percent_diam)
        # Attention cela retourne un dictionnaire avec aa['hdf5_id'] = value
        hdf_tiges_seuil_offset = h5store.get_postprocessing(hdf5file, 'Tiges_seuil_offset')
        hdf_tiges_percent_diam = h5store.get_postprocessing(hdf5file, 'Tiges_percent_diam')

        # Il faut convertir les id hdf en position des tiges qui correspondent a base_tiges
        Tiges_seuil_offset = {}
        Tiges_percent_diam = {}
        for i in range(len(base_tiges)):

            hdf5_tigeid = get_hdf5_tigeid(hdf5file, i)
            if hdf5_tigeid in hdf_tiges_seuil_offset:
                # On regarde si la valeur enregistrée est pas None et
                # si c'est le cas on met la valeur de l'offset à 0
                tmp_offset = hdf_tiges_seuil_offset[hdf5_tigeid]
                if tmp_offset is None:
                    tmp_offset = 0
                Tiges_seuil_offset[i] = tmp_offset
            else:
                Tiges_seuil_offset[i] = 0
                try:
                    # Mise a jour des données de la tiges dans le fichier hdf5
                    new_data_tige = {"Tiges_seuil_offset": Tiges_seuil_offset[i]}
                    h5store.save_tige_dict_to_hdf(hdf5file, hdf5_tigeid, new_data_tige)
                except Exception as e:
                    print("La mise à jour des options a raté")
                    print(e)

            if hdf5_tigeid in hdf_tiges_percent_diam:
                # On regarde si la valeur enregistrée est pas None et
                # si c'est le cas on met la valeur par défault de percent_diam
                tmp_percent_diam = hdf_tiges_percent_diam[hdf5_tigeid]
                if tmp_percent_diam is None:
                    tmp_percent_diam = PERCENT_DIAM_DEFAULT
                Tiges_percent_diam[i] = tmp_percent_diam
            else:
                Tiges_percent_diam[i] = PERCENT_DIAM_DEFAULT
                try:
                    # Mise a jour des données de la tiges dans le fichier hdf5
                    new_data_tige = {'Tiges_percent_diam': Tiges_percent_diam[i]}
                    h5store.save_tige_dict_to_hdf(hdf5file, hdf5_tigeid, new_data_tige)
                except Exception as e:
                    print("La mise à jour des options a raté")
                    print(e)

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
                                         pas=detection_step,
                                         tiges_seuil_offset=Tiges_seuil_offset,
                                         tiges_percent_diam=Tiges_percent_diam,
                                         output_function=none_print)

                tiges_x = traite_tiges2(pre_data[0]['tiges_data'], pas=detection_step)[0]
                #print(tiges_tailles/0.3)
                #print(tiges_x.shape)
                max_array_size = tiges_x.shape[2] + 100
                print("Maximum size for iterations: %i" % max_array_size)

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
                                            'pas': detection_step,
                                            'outputdata':data_out,
                                            'tiges_seuil_offset': Tiges_seuil_offset,
                                            'tiges_percent_diam': Tiges_percent_diam,
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
    if not tiges_data:  # if no tige has been processed
        return

    # Retrieve the detection step from the h5 file
    detection_step = h5store.get_detection_step(hdf5file)

    if detection_step is None:
        detection_step = DETECTION_STEP_DEFAULT

    tiges_x, tiges_y, tiges_s, tiges_tailles, tiges_angles, tiges_tip_angles, \
            tiges_lines, tiges_measure_zone = traite_tiges2(
                tiges_data, pas=detection_step, return_estimation_zone=True)

    if save_to_hdf5:
        for id_tige in range(len(tiges_x)):
            data_tmp = {"smooth_xc": tiges_x[id_tige],
                        "smooth_yc": tiges_y[id_tige],
                        "smooth_s": tiges_s[id_tige],
                        "taille": tiges_tailles[id_tige],
                        "angle": tiges_angles[id_tige],
                        "angle_au_bout": tiges_tip_angles[id_tige],
                        "angle_zone_de_mesure": tiges_measure_zone[id_tige]}

            tigeidhdf = get_hdf5_tigeid(hdf5file, id_tige)
            h5store.save_tige_dict_to_hdf(hdf5file, tigeidhdf, data_tmp)
    else:
        return (tiges_x, tiges_y, tiges_s, tiges_tailles, tiges_angles,
                tiges_tip_angles, tiges_lines, tiges_measure_zone)


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
    tige_s = h5store.get_postprocessing(hdf5file, 'smooth_s', hdf_tige_id)
    tige_taille = h5store.get_postprocessing(hdf5file, 'taille', hdf_tige_id)
    tige_angle = h5store.get_postprocessing(hdf5file, 'angle', hdf_tige_id)
    tige_tip_angle = h5store.get_postprocessing(hdf5file, 'angle_au_bout', hdf_tige_id)
    tige_measure_zone = h5store.get_postprocessing(hdf5file, 'angle_zone_de_mesure',
                                                   hdf_tige_id).tolist()

    print("Chargement fini!")

    # Masque les données invalide qui ont une valeur de -30000
    badval = -30000
    tige_x = ma.masked_equal(tige_x, badval)
    tige_y = ma.masked_equal(tige_y, badval)
    tige_s = ma.masked_equal(tige_s, badval)
    tige_taille = ma.masked_equal(tige_taille, badval)
    tige_angle = ma.masked_equal(tige_angle, badval)
    tige_tip_angle = ma.masked_equal(tige_tip_angle, badval)

    # Creation d'une liste de la ligne du centre de la tige compatible
    # avec matplotlib LineCollection pour afficher le squelette de la
    # tige au cours du temps
    # Version très lente l'ancienne version
    # tige_lines = [zip(tige_x[t], tige_y[t]) for t in range(len(tige_x))]

    # Version beaucoup plus rapide
    tige_lines = ma.dstack((tige_x, tige_y))

    return (tige_x, tige_y, tige_s, tige_taille, tige_angle, tige_tip_angle,
            tige_measure_zone, tige_lines)

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

    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Récupération des ids dans le fichier hdf5
    tigeid = get_hdf5_tigeid(hdf5file, cur_tige)
    tige_name = h5store.get_tiges_names(hdf5file, tigeid)

    # Suppression de la tige dans le fichier hdf5
    h5store.delete_tige(hdf5file, tigeid)

    # remet a zero les tracés sur la figure
    reset_graph_data()

    # recharger les données du fichier hdf5 en mémoire
    load_hdf5_tiges(hdf5file)

    interekt.plot_image(cur_image, force_clear=True)


###################### Fenetre pour afficher une tige #########################
def export_one_tige():
    """
    Fonction pour exporter les données d'une tige (celle définit par
    la variable globale cur_tige) pour le pas de temps affiché à
    l'écran en csv
    """

    detection_step = h5store.get_detection_step(hdf5file)

    if detection_step is None:
        detection_step = DETECTION_STEP_DEFAULT

    # Id de la tige dans le fichier h5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Le nom de la tige
    tige_name = h5store.get_tiges_names(hdf5file, hdf_tige_id)
    proposed_filename = "tige_%s_image_%i.csv"%(tige_name, cur_image+1)
    start_title = _("Export data from organ")
    outfileName = filedialog.asksaveasfilename(
            parent=toptige,
            filetypes=[("Comma Separated Value, csv","*.csv")],
            title=start_title + " %s"%tige_name,
            initialfile=proposed_filename,
            initialdir=base_dir_path)

    if len(outfileName) > 0:
        #Creation du tableau avec pandas
        tx = tiges_data.xc[cur_tige,cur_image]
        ty = tiges_data.yc[cur_tige,cur_image]
        tsmoothx, tsmoothy, tangle, ts, tW = traite_tige2(
                tx, ty, tiges_data.diam[cur_tige, cur_image]/2.0, pas=detection_step)
        tsmoothx = tsmoothx[:-1]
        tsmoothy = tsmoothy[:-1]
        tcourbure = diff(tangle)/diff(ts)
        tcourbure = hstack((tcourbure,tcourbure[-1]))
        data_tmp = {'tige':[tige_name]*len(tsmoothx),'image':[cur_image+1]*len(tsmoothy),
                    'angle (deg)': tangle,'x (pix)': tsmoothx,'y (pix)': tsmoothy,
                    'abscisse curviligne (pix)': ts, 'courbure c (deg/pix)':tcourbure,
                    'angle_0_360 (pix)': convert_angle(tangle)}

        data_tmp = pd.DataFrame(data_tmp, columns=['tige','image','angle (deg)','x (pix)',
                                                   'y (pix)', 'abscisse curviligne (pix)',
                                                   'courbure c (deg/pix)','angle_0_360 (pix)'])

        data_tmp.to_csv(outfileName, index=False)


def save_tige_options():
    """
    Fonction pour sauvegarder les options modifiées dans le gui
    (fonction show_tiges_options) dans le fichier hdf5
    """
    reset_graph_data()

    new_tige_name = tktigeid.get()
    new_offset = tk_tige_offset.get()
    new_percent_diam = tk_tige_percent_diam.get()
    hdf5_tigeid = get_hdf5_tigeid(hdf5file, cur_tige)

    try:
        h5store.save_tige_name(hdf5file, hdf5_tigeid, new_tige_name)
        # Mise a jour des données de la tiges dans le fichier hdf5
        new_data_tige = {"Tiges_seuil_offset": new_offset,
                         'Tiges_percent_diam': new_percent_diam}
        #besoin de recup l'id de la tiges dans le fichier hdf5
        h5store.save_tige_dict_to_hdf(hdf5file, hdf5_tigeid, new_data_tige)
    except Exception as e:
        print("La mise à jour des options a raté")
        print(e)

    # Rechargement des données des tiges a partir du hdf5
    load_hdf5_tiges(hdf5file)

    # Replot the main image
    interekt.plot_image(cur_image, keep_zoom=False, force_clear=True)


def show_tige_options():
    global toptige, tktigeid, tk_tige_offset, tk_tige_percent_diam

    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Récupération des données du fichier hdf5 et de la variable tige_id_mapper
    tige_name = tiges_names[cur_tige]
    hdf5_tigeid = get_hdf5_tigeid(hdf5file, cur_tige)
    seuil = h5store.get_postprocessing(hdf5file, 'Tiges_seuil_offset', hdf5_tigeid)
    percent_diam = h5store.get_postprocessing(hdf5file, 'Tiges_percent_diam', hdf5_tigeid)

    # Ajout d'une boite tk pour l'export
    toptige = Tk.Toplevel(master=root)
    start_title = _("Settings for organ")
    if python2:
        toptige.title(start_title.decode("utf8") + " %i"%hdf5_tigeid)
    else:
        toptige.title(start_title + " %i"%hdf5_tigeid)

    #Case pour changer nom de la tige
    idframe = Tk.Frame(toptige)
    Tk.Label(idframe, text=_("Organ name:")).pack(fill='x', expand=True)
    tktigeid = Tk.Entry(idframe)
    tktigeid.insert(0, str(tige_name))
    tktigeid.pack(fill='x', expand=True)

    Tk.Label(idframe, text=_("Threshold sensitivity:")).pack(fill='x', expand=True)
    tk_tige_offset = Tk.DoubleVar()

    if seuil is not None:
        tk_tige_offset.set(seuil)

    w2 = Tk.Scale(idframe, from_=-5, to=5, resolution=0.1, variable=tk_tige_offset,
                  orient=Tk.HORIZONTAL)
    w2.pack(fill='x', expand=True)

    Tk.Label(idframe, text=_("Diameter detection range:")).pack(fill='x', expand=True)
    tk_tige_percent_diam = Tk.DoubleVar()

    if percent_diam is not None:
        tk_tige_percent_diam.set(percent_diam)
    else:
        tk_tige_percent_diam.set(PERCENT_DIAM_DEFAULT)

    w3 = Tk.Scale(idframe, from_=0, to=10, resolution=0.1, variable=tk_tige_percent_diam,
                  orient=Tk.HORIZONTAL)
    w3.pack(fill='x', expand=True)

    Tk.Button(idframe, text=_("Save"),
              command=save_tige_options).pack(fill='x', expand=True)

    idframe.pack(fill='x', expand=True)

    start_text = _("Export organ at")
    end_text = _("to csv.")
    if python2:
        start_text = start_text.decode('utf8')
        end_text = end_text.decode('utf8')
    tigebutton_export = Tk.Button(master=toptige,
                                  text=start_text+" t=%i "%(cur_image+1)+end_text,
                                  command=export_one_tige)

    tigebutton_export.pack(side=Tk.BOTTOM)


tktigeid = None
def show_time_series(tige_id=None):
    global cur_tige
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    if tige_id is not None:
        cur_tige = int(tige_id)

    # Retrieving the tige id from the hdf5 file
    tige_hdf_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Retrieving useful data as time-indexed arrays
    radii = tiges_data.diam[cur_tige].mean(axis=1)   # space-averaged tige radii
    x, y, s, lengths, angles, tip_angles, measure_zone, lines = \
            load_postprocessed_data(hdf5file, tige_hdf_id)

    # Create the figure with the tige name
    tige_name = h5store.get_tiges_names(hdf5file, tige_hdf_id)
    fig_time = mpl.figure(strings["Organ"] + ' %s'%str(tige_name), figsize=(10, 6))

#     grid = mpl.GridSpec(4, 3, wspace=.4, hspace=.4)
    grid = mpl.GridSpec(6, 3)

    # Scaling
    length_unit = "pix"
    scale_cmpix = h5store.get_pixelscale(hdf5file)
    if scale_cmpix is not None:
        length_unit = "cm"
        x *= scale_cmpix
        y *= scale_cmpix
        s *= scale_cmpix
        lengths *= scale_cmpix
        radii *= scale_cmpix
        lines *= scale_cmpix

    # if a steady-state image has been defined
    if interekt.steady_state_image is not None:
        steady_state_line = lines[interekt.steady_state_image]

    times, time_units, time_label, graduation, excluded_image, mask = get_time_data()
    cur_time = times[cur_image]

    if excluded_image is not None:  # steady-state image excluded from time series
        times = times[mask]
        lengths = lengths[mask]
        radii = radii[mask]
        angles = angles[mask]
        tip_angles = tip_angles[mask]
        lines = lines[mask]

    # Subplot with tige profiles
    axProfiles = fig_time.add_subplot(grid[:,0])
    lcollec = LineCollection(lines, linewidth=(2,), cmap='viridis')
    lcollec.set_array(times)
    axProfiles.add_collection(lcollec)
    axProfiles.set_xlim((x.min(), x.max()))
    axProfiles.set_ylim((y.min(), y.max()))
    axProfiles.set_xlabel("x - x0 (%s)"%length_unit)
    axProfiles.set_ylabel("y - y0 (%s)"%length_unit)
    axProfiles.axis('equal')
    if interekt.steady_state_image is not None:
        # The steady-state profile is colored in red
        lcollec_ss = LineCollection([steady_state_line], linewidth=(2,), color='red')
        axProfiles.add_collection(lcollec_ss)

    axAngle = fig_time.add_subplot(grid[:2, 1:])
    axAngle.grid(True)
    axAngle.xaxis.set_major_locator(MultipleLocator(graduation))
    axAngle.set_ylabel(strings["tip angle"] + " " + strings["degree"])

    if interekt.angle_0_360.get():
        tip_angles = convert_angle(tip_angles)

    axAngle.plot(times, tip_angles, '+-', color=tiges_colors[cur_tige], lw=2)

    axLength = fig_time.add_subplot(grid[2:4, 1:], sharex=axAngle)
    axLength.grid(True)
    axLength.xaxis.set_major_locator(MultipleLocator(graduation))
    axLength.set_ylabel(strings["length"] + " (%s)"%length_unit)
    axLength.plot(times, lengths, '+-', color=tiges_colors[cur_tige], lw=2)

    axRadius = fig_time.add_subplot(grid[4:6, 1:], sharex=axAngle)
    axRadius.grid(True)
    axRadius.xaxis.set_major_locator(MultipleLocator(graduation))
    axRadius.set_xlabel(time_label)
    axRadius.set_ylabel(strings["radius"] + " (%s)"%length_unit)
    axRadius.plot(times, radii, '+-', color=tiges_colors[cur_tige], lw=2)

    if excluded_image == cur_image:
        # If the steady-state image has to be excluded from time series and is the current
        # image, just create empty Line2D.
        colortige, = axProfiles.plot([], [], 'k', lw=2.5)
        time_select_angle, = axAngle.plot([None, None], axAngle.get_ylim(), 'k--', lw=1.5)
        time_select_length, = axLength.plot([None, None], axLength.get_ylim(),
                                            'k--', lw=1.5)
        time_select_radius, = axRadius.plot([None, None], axRadius.get_ylim(),
                                            'k--', lw=1.5)
    else:
        # Otherwise, make the current tige profile black and thicker...
        xt = x[cur_image, ~x[cur_image].mask]
        yt = y[cur_image, ~y[cur_image].mask]
        colortige, = axProfiles.plot(xt, yt, 'k', lw=2.5)
        # ...and mark the current time as a vertical dotted line in time series.
        time_select_angle, = axAngle.plot([cur_time, cur_time], axAngle.get_ylim(),
                                          'k--', lw=1.5)
        time_select_length, = axLength.plot([cur_time, cur_time], axLength.get_ylim(),
                                            'k--', lw=1.5)
        time_select_radius, = axRadius.plot([cur_time, cur_time], axRadius.get_ylim(),
                                            'k--', lw=1.5)

    def on_press(event):

        if event.xdata is not None and (axAngle.contains(event)[0]
                                        or axLength.contains(event)[0]
                                        or axRadius.contains(event)[0]):

            # image closest to clicked point
            image = ((times - event.xdata)**2).argmin()
            xt = x[image, ~x[image].mask]
            yt = y[image, ~y[image].mask]
            try:
                colortige.set_data(xt, yt)
            except Exception as e:
                print(e)

            time_select_angle.set_xdata([times[image], times[image]])
            time_select_length.set_xdata([times[image], times[image]])
            time_select_radius.set_xdata([times[image], times[image]])

            # redraw figure
            fig_time.canvas.draw()

            # Change figure on main frame
            interekt.plot_image(image, keep_zoom=True)

    def on_close(event):
        mpl.close(fig_time)

    # Add events for the figure
    fig_time.canvas.mpl_connect('button_press_event', on_press)
    fig_time.canvas.mpl_connect('close_event', on_close)

    grid.tight_layout(fig_time)

    # Show the figure
    fig_time.show()


def show_heatmaps():
    """
    Displays the spatiotemporal variations of geometric variables as heatmaps.

    These variables are:
      - the curvature C
      - the radius R
      - the product CR
    """
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Retrieving the tige id from the hdf5 file
    tige_hdf_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Retrieving useful data as spacetime 2D-arrays
    radii = tiges_data.diam[cur_tige]
    s, angles = load_postprocessed_data(hdf5file, tige_hdf_id)[2:5:2]

    # Scaling
    length_unit, inv_length_unit = "pix", r"pix$^{-1}$"
    scale_cmpix = h5store.get_pixelscale(hdf5file)
    if scale_cmpix is not None:
        length_unit, inv_length_unit = "cm", r"cm$^{-1}$"
        s *= scale_cmpix

    times, time_units, time_label, graduation, excluded_image, mask = get_time_data()
    cur_time = times[cur_image]

    # If the steady-state image has to be excluded from time series
    # (for instance because it has been taken a very long time after other images)
    if excluded_image is not None:
        times = times[mask]
        s = s[mask]
        radii = radii[mask]
        angles = angles[mask]

    # Retrieve the detection step from the h5 file
    detection_step = h5store.get_detection_step(hdf5file)
    if detection_step is None:
        detection_step = DETECTION_STEP_DEFAULT

    # Size of the averaging window
    W = int(4 * round(radii.mean() / detection_step))

    # Smoothing and computation of the curvatures
    smooth_s = zeros_like(angles) - 3000
    C = zeros_like(angles) - 3000  # curvatures
    R = zeros_like(angles) - 3000  # radii
    cutoff_left, cutoff_right = 2, 2
    for i in range(len(angles)):
        imax = flatnonzero(angles[i].mask == False).max() + 1
        angles_i = angles[i, cutoff_left:imax-cutoff_right]
        s_i = s[i, cutoff_left:imax-cutoff_right]
        R_i = radii[i, cutoff_left:imax-cutoff_right]
        C_i, smooth_angles_i = get_tige_curvature(angles_i, s_i,
                                                  smoothing=True, window_width=W)
        smooth_s[i, :len(s_i)] = s_i
        C[i, :len(C_i)] = C_i
        R[i, :len(R_i)] = R_i

    smooth_s = ma.masked_equal(smooth_s, -3000)
    smooth_s = ma.masked_invalid(smooth_s)
    C = ma.masked_equal(C, -3000)
    C = ma.masked_invalid(C)
    R = ma.masked_equal(R, -3000)
    R = ma.masked_invalid(R)

    imax = flatnonzero(smooth_s[-1].mask == False).max() + 1
    smooth_s = smooth_s[:,:imax]
    C = C[:,:imax]
    R = R[:,:imax]
    CR = C * R

    t_min, t_max = times.min(), times.max()
    s_max = smooth_s.max()
    C_min, C_max = C.min(), C.max()
    R_min, R_max = R.min(), R.max()

    # Create the figure with the tige name
    tige_name = h5store.get_tiges_names(hdf5file, tige_hdf_id)
    start_title = strings["Spatiotemporal heatmaps"] + " " + strings["for organ"]
    fig_heatmaps = mpl.figure(start_title + " %s"%(tige_name), figsize=(12, 10))

    grid = mpl.GridSpec(3, 1)

    cmap = mpl.get_cmap('jet')

    # Curvature heatmap
    axC = fig_heatmaps.add_subplot(grid[0,:])

    axC.set_xlim([-s_max/100, s_max + s_max/100])
    axC.set_ylim([-t_max/50, t_max + t_max/50])
    axC.set_xlabel('s (%s)'%length_unit)
    axC.set_ylabel(time_label)

    linear_max = quantile(abs(C), 0.99)
    cmeshC = axC.pcolormesh(smooth_s[-1], times, C,
                            norm=SymLogNorm(linthresh=linear_max, linscale=4),
                            cmap=cmap, rasterized=True)

    cbarC = mpl.colorbar(cmeshC, use_gridspec=True, pad=0.01, extend='both')
    cbarC.ax.tick_params(labelsize=8)
    cbarC.ax.get_yaxis().labelpad = 15
    cbarC.ax.set_ylabel(strings["curvature"] + " (%s)"%inv_length_unit, rotation=270)

    if excluded_image == cur_image:
        # If the steady-state image has to be excluded from time series and is the current
        # image, just create empty Line2D.
        time_select, = axC.plot(axC.get_xlim(), [None, None], 'k--', lw=2)
    else:
        # Otherwise mark the current time as a horizontal dotted line.
        time_select, = axC.plot(axC.get_xlim(), [cur_time]*2, 'k--', lw=2)

    def on_press(event):
        if event.ydata is not None and axC.contains(event)[0]:
            # image closest to clicked point
            image = ((times - event.ydata)**2).argmin()
            time_select.set_ydata([times[image], times[image]])
            # redraw figure
            fig_heatmaps.canvas.draw()
            # Change figure on main frame
            interekt.plot_image(image, keep_zoom=True)

    # Radius heatmap
    axR = fig_heatmaps.add_subplot(grid[1,:])

    axR.set_xlim([-s_max/100, s_max + s_max/100])
    axR.set_ylim([-t_max/50, t_max + t_max/50])
    axR.set_ylabel(time_label)

    cmeshR = axR.pcolormesh(smooth_s[-1], times, R,
                            cmap=cmap, rasterized=True)

    cbarR = mpl.colorbar(cmeshR, use_gridspec=True, pad=0.01, extend='both')
    cbarR.ax.tick_params(labelsize=8)
    cbarR.ax.get_yaxis().labelpad = 15
    cbarR.ax.set_ylabel(strings["radius"] + " (%s)"%length_unit, rotation=270)

    # Curvature × Radius heatmap
    axCR = fig_heatmaps.add_subplot(grid[2,:])

    axCR.set_xlim([-s_max/100, s_max + s_max/100])
    axCR.set_ylim([-t_max/50, t_max + t_max/50])
    axCR.set_xlabel(strings["s"] + ' (%s)'%length_unit)
    axCR.set_ylabel(time_label)

    linear_max = quantile(abs(CR), 0.99)
    cmeshCR = axCR.pcolormesh(smooth_s[-1], times, CR,
                              norm=SymLogNorm(linthresh=linear_max, linscale=4),
                              cmap=cmap, rasterized=True)

    cbarCR = mpl.colorbar(cmeshCR, use_gridspec=True, pad=0.01, extend='both')
    cbarCR.ax.tick_params(labelsize=8)
    cbarCR.ax.get_yaxis().labelpad = 15
    cbarCR.ax.set_ylabel(strings["curvature"] + " × " + strings["radius"], rotation=270)

    fig_heatmaps.canvas.mpl_connect('button_press_event', on_press)
    grid.tight_layout(fig_heatmaps)
    fig_heatmaps.show()


def show_angle_and_curvature():
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Retrieve the detection step from the h5 file
    detection_step = h5store.get_detection_step(hdf5file)

    if detection_step is None:
        detection_step = DETECTION_STEP_DEFAULT

    # Retrieving the tige id from the hdf5 file
    tige_hdf_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Retrieving useful data ('d' is for 'dummy')
    d, d, s, d, angles, d, d, lines = load_postprocessed_data(hdf5file, tige_hdf_id)

    # Create the figure with the tige name
    tige_name = h5store.get_tiges_names(hdf5file, tige_hdf_id)
    start_title = _("Angles and curvatures") + " " + strings["for organ"]
    fig_curvature = mpl.figure(start_title + " %s"%(tige_name), figsize=(12, 10))

    # Scaling
    scale_cmpix = h5store.get_pixelscale(hdf5file)
    unit, inv_unit = "(pix)", r"(pix$^{-1}$)"
    nb_averaging_points = 80  # default
    if scale_cmpix is not None:
        unit, inv_unit = "(cm)", r"(cm$^{-1}$)"
        s *= scale_cmpix
        lines *= scale_cmpix
        nb_averaging_points = int(round(CURVATURE_AVERAGING_LENGTH
                                        / (scale_cmpix * detection_step)))

    # Size of the averaging window
    W = int(4 * round(tiges_data.diam[cur_tige].mean() / detection_step))

    # Time array and unit
    times, time_units = get_time_data()[:2]
    time_unit = time_units["time unit"]

#     grid = mpl.GridSpec(3, 2, wspace=.7, hspace=.7)
    grid = mpl.GridSpec(3, 2)

    axProfiles = fig_curvature.add_subplot(grid[0,:])
    axAngle = fig_curvature.add_subplot(grid[1,:])
    axCurvature = fig_curvature.add_subplot(grid[2,:], sharex=axAngle)

    axProfiles.set_xlabel(r"$x - x_0$ %s"%unit)
    axProfiles.set_ylabel(r"$y - y_0$ %s"%unit)

    axAngle.set_xlabel(strings["s"] + " %s"%unit)
    axAngle.set_ylabel(strings["angle"] + " " + strings["degree"])
    axAngle.grid(True)

    axCurvature.set_xlabel(strings["s"] + " %s"%unit)
    axCurvature.set_ylabel(strings["curvature"] + " %s"%inv_unit)
    axCurvature.grid(True)

    for mark in marks:
        image, name = mark["image"], mark["name"]
        mark_index = mark["intersection_index"]
        time = times[image]
        if time_unit == strings["image"]:
            mark_label = "mark " + str(name) + " (" + strings["image"] + " %i)"%image
        else:  # if there is a proper time unit
            mark_label = "mark " + str(name) + " (t = %.1f "%time + time_unit + ")"

        s_i, angles_i, lines_i = s[image], angles[image], lines[image]
        x, y = lines_i[:, 0], lines_i[:, 1]

        # Remove non-valid values
        imax = flatnonzero(angles_i.mask == False).max() + 1
        angles_i = angles_i[:imax]
        s_i = s_i[:imax]
        x = x[:imax]
        y = y[:imax]

        curvatures, smooth_angles = get_tige_curvature(angles_i, s_i,
                                                       smoothing=True, window_width=W)
        smooth_angles = rad2deg(smooth_angles)
        if interekt.angle_0_360.get():
            angles = convert_angle(angles)
            smooth_angles = convert_angle(smooth_angles)

        l, = axAngle.plot(s_i, smooth_angles, '-', label=mark_label)
        color = l.get_color()
        # Big point where the mark is placed
        axAngle.plot(s_i[mark_index], smooth_angles[mark_index], 'o', color=color, ms=4)

        axCurvature.plot(s_i, curvatures, '--', color=color, label=mark_label)
        # Big point where the mark is placed
        axCurvature.plot(s_i[mark_index], curvatures[mark_index],
                         'o', color=color, ms=4)
        # The averaging zone (used for computing mean curvature) as a continuous line
        averaging_zone = slice(mark_index, mark_index+nb_averaging_points)
        axCurvature.plot(s_i[averaging_zone], curvatures[averaging_zone], '-',
                         color=color)

        profile = mpl.Line2D(x, y)
        profile.set_linestyle("-")
        profile.set_linewidth(1)
        profile.set_color(color)
        axProfiles.add_line(profile)
        # Big point where the mark is placed
        xy_mark = mark["intersection_point_from_base"]
        axProfiles.plot(xy_mark[0], xy_mark[1], 'o', color=color, ms=4)

        # The averaging zone (used for computing mean curvature) as a thicker line
        averaging_line = mpl.Line2D(x[averaging_zone], y[averaging_zone])
        averaging_line.set_linestyle("-")
        averaging_line.set_linewidth(2)
        averaging_line.set_color(color)
        axProfiles.add_line(averaging_line)

    if cur_image not in [mark["image"] for mark in marks]:
        if time_unit == strings["image"]:
            name = strings["image"] + " %i"%cur_image  # image number
        else:  # if there is a proper time unit
            time = times[cur_image]
            name = "t = %.1f "%time + time_unit
        s_i, angles_i, lines_i = s[cur_image], angles[cur_image], lines[cur_image]
        x, y = lines_i[:, 0], lines_i[:, 1]
        # Remove non-valid values
        imax = flatnonzero(angles_i.mask == False).max() + 1
        angles_i = angles_i[:imax]
        s_i = s_i[:imax]
        x = x[:imax]
        y = y[:imax]
        curvatures, smooth_angles = get_tige_curvature(angles_i, s_i,
                                                       smoothing=True, window_width=W)
        smooth_angles = rad2deg(smooth_angles)
        if interekt.angle_0_360.get():
            angles_i = convert_angle(angles_i)
            smooth_angles = convert_angle(smooth_angles)
        l, = axAngle.plot(s_i, smooth_angles, '-', lw=1, label=name)
        color = l.get_color()
        axCurvature.plot(s_i, curvatures, '-', color=color, lw=1, label=name)
        profile = mpl.Line2D(x, y)
        profile.set_linestyle("-")
        profile.set_linewidth(1)
        profile.set_color(color)
        axProfiles.add_line(profile)

    axProfiles.axis('equal')
    axAngle.legend()
    axCurvature.legend()

    def on_close(event):
        mpl.close(fig_curvature)

    # Add events for the figure
    fig_curvature.canvas.mpl_connect('close_event', on_close)

    grid.tight_layout(fig_curvature)
    fig_curvature.show()


def show_growth_rate():
    """Interactive estimation of the growth rate."""
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Recup l'id de la tige dans le fichier h5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Récuperation des données dont on a besoin
    x, y, d, lengths, d, d, d, lines = load_postprocessed_data(hdf5file, hdf_tige_id)
    # 'd' stands for 'dummy'

    # Scaling factor (conversion from pixels to cm)
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Scaling
    length_unit, inv_length_unit = "pix", r"pix$^{-1}$"
    if scale_cmpix is not None:
        length_unit, inv_length_unit = "cm", r"cm$^{-1}$"
        x *= scale_cmpix
        y *= scale_cmpix
        lengths *= scale_cmpix
        lines *= scale_cmpix

    # if a steady-state image has been defined
    if interekt.steady_state_image is not None:
        steady_state_line = lines[interekt.steady_state_image]

    times, time_units, time_label, graduation, excluded_image, mask = get_time_data()
    cur_time = times[cur_image]

    if excluded_image is not None:  # steady-state image excluded from time series
        times = times[mask]
        lengths = lengths[mask]
        lines = lines[mask]

    t_min, t_max = times.min(), times.max()

    # Creating the figure
    start_title = _("Estimation of the growth rate for organ")
    fig_growth_rate = mpl.figure(start_title + " %i"%(cur_tige+1), figsize=(12, 10))

    grid = mpl.GridSpec(3, 3)

    # Subplot with tige profiles
    axProfiles = fig_growth_rate.add_subplot(grid[:, 0])
    lcollec = LineCollection(lines, linewidth=(2,), cmap='viridis')
    lcollec.set_array(times)
    axProfiles.add_collection(lcollec)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    axProfiles.set_xlim((xmin, xmax))
    axProfiles.set_ylim((ymin, ymax))
    axProfiles.set_xlabel("x - x0 (%s)"%length_unit)
    axProfiles.set_ylabel("y - y0 (%s)"%length_unit)
    axProfiles.axis('equal')
    if interekt.steady_state_image is not None:
        # The steady-state profile is colored in red
        lcollec_ss = LineCollection([steady_state_line], linewidth=(2,), color='red')
        axProfiles.add_collection(lcollec_ss)

    text_growth = axProfiles.text(xmin + (xmax - xmin) / 20, ymax, '')

    axSelect = fig_growth_rate.add_subplot(grid[0, 1:])
    axSelect.plot(times, lengths, '+-', color=tiges_colors[cur_tige], lw=2)
    axSelect.set_xlabel(time_label)
    axSelect.set_ylabel(strings["length"] + " (%s)"%length_unit)
    axSelect.grid(True)
    axSelect.xaxis.set_major_locator(MultipleLocator(graduation))

    axFit = fig_growth_rate.add_subplot(grid[1, 1:])
    axFit.plot(times, lengths, '+-', color=tiges_colors[cur_tige], lw=2)
    axFit.set_xlabel(time_label)
    axFit.set_ylabel(strings["length"] + " (%s)"%length_unit)
    axFit.grid(True)
    axFit.xaxis.set_major_locator(MultipleLocator(graduation))

    axResiduals = fig_growth_rate.add_subplot(grid[2, 1:])
    axResiduals.set_xlim((t_min, t_max))
    axResiduals.set_xlabel(time_label)
    axResiduals.set_ylabel(strings["residuals"])
    axResiduals.grid(True)
    axResiduals.xaxis.set_major_locator(MultipleLocator(graduation))
    axResiduals.plot([times[0], times[-1]], [0, 0], 'k-', lw=2)
    residuals_plot, = axResiduals.plot([], [], 'k+')

    def compute_growth(P1, P2, plot_residuals=False):
        """Compute growth and RER from two points on the growth curve."""
        t1, L1 = P1.get_data()
        t2, L2 = P2.get_data()
        t1, t2, L1, L2 = float(t1), float(t2), float(L1), float(L2)
        if t1 > t2:
            t1, t2 = t2, t1
            L1, L2 = L2, L1
        dLdt = (L2 - L1) / (t2 - t1)
        lengths_fitted = dLdt * (times - t1) + L1

        # create unit strings
        time_unit = time_units["time unit"]
        RER_unit, RER_unit_TeX = time_units["RER unit"], time_units["RER unit TeX"]
        dLdt_unit = "%s.%s"%(length_unit, RER_unit)
        dLdt_unit_TeX = "%s.%s"%(length_unit, RER_unit_TeX)

        if time_unit == strings["min"]:
            # The Relative Elongation Rate (RER) should be in h¯¹.
            # Hence the multiplication by 60.
            dLdt *= 60
        elif time_unit == strings["days"]:
            dLdt /= 24

        text = (r"""$\frac{dL}{dt} = %0.4f$ %s"""
                """\n"""%(dLdt, dLdt_unit_TeX))
        text_growth.set_text(text)

        # look for the indices i1 and i2 associated with time t1 and t2
        if t1 < t_min or t1 > t_max:  # out of bounds
            i1 = "out"
        else:
            i1 = flatnonzero(times >= t1)[0]
        if t2 < t_min or t2 > t_max:  # out of bounds
            i2 = "out"
        else:
            i2 = flatnonzero(times >= t2)[0]

        growth_data = {"L1": L1, "L2": L2, "t1": t1, "t2": t2, "i1": i1, "i2": i2,
                       "dLdt": dLdt, "length_unit": length_unit,
                       "dLdt_unit": dLdt_unit, "dLdt_unit_TeX": dLdt_unit_TeX,
                       "RER_unit": RER_unit, "RER_unit_TeX": RER_unit_TeX}

        # Saving growth_data in the hdf5 file
        h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id, {'growth_data': growth_data})

        if plot_residuals:
            if i1 == "out":
                i1 = 0
            if i2 == "out":
                i2 = -1
            residuals = lengths[i1:i2] - lengths_fitted[i1:i2]
            y_half_span = max(abs(residuals.min()), abs(residuals.max()))
            axResiduals.set_ylim((-y_half_span, y_half_span))
            residuals_plot.set_data(times[i1:i2], residuals)

        fig_growth_rate.canvas.draw()

        return dLdt

    growth_data = h5store.get_postprocessing(hdf5file, 'growth_data', hdf_tige_id)
    if growth_data is not None:
        t1, t2 = growth_data['t1'], growth_data['t2']
        t1, t2 = max(t1, t_min), min(t2, t_max)
        L1, L2 = growth_data['L1'], growth_data['L2']
    else:
        t1, t2 = t_min, t_max
        L1, L2 = lengths[0], lengths[-1]

    # In case there are NotANumbers in lengths
    if isnan(L1):
        L1 = min(lengths)
    if isnan(L2):
        L2 = max(lengths)

    # The fitting is manual, using draggable points
    P1, = axFit.plot(t1, L1, 'ro', ms=12, picker=5)
    P2, = axFit.plot(t2, L2, 'ro', ms=12, picker=5)
    dpoints = DraggablePoints([P1, P2], linked=True)
    compute_growth(P1, P2, plot_residuals=True)

    def on_pick(evt):
        dpoints.on_press(evt)
        zone_selector.set_visible(True)
        zone_selector.update()

    def on_motion(evt):
        if evt.inaxes == axFit:
            dpoints.on_motion(evt)

    def on_release(evt):
        if evt.inaxes == axFit:
            dpoints.on_release(evt)
            P1, P2 = dpoints.points[:2]
            compute_growth(P1, P2, plot_residuals=True)
            zone_selector.set_visible(True)
            zone_selector.update()

    def on_close(evt):
        mpl.close(fig_growth_rate)

    fig_growth_rate.canvas.mpl_connect('pick_event', on_pick)
    fig_growth_rate.canvas.mpl_connect('button_release_event', on_release)
    fig_growth_rate.canvas.mpl_connect('motion_notify_event', on_motion)
    fig_growth_rate.canvas.mpl_connect('close_event', on_close)

    # A zone for fitting is selected using a RectangleSelector widget

    def on_select(eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        zone_selector.update()
        xmin, xmax, ymin, ymax = zone_selector.extents
        # Reset axes limits (i.e. zoom in)
        axFit.set_xlim((xmin, xmax))
        axFit.set_ylim((ymin, ymax))
        axResiduals.set_xlim((xmin, xmax))
        # If a draggable point is outside the rectangle selector,
        # the point has to be moved.
        reinit = False
        P1, P2 = dpoints.points
        x_P1, y_P1 = P1.get_data()
        x_P2, y_P2 = P2.get_data()
        x_shift, y_shift = (xmax - xmin) / 100, (ymax - ymin) / 100
        if x_P1 < xmin or x_P1 > xmax or y_P1 < ymin or y_P1 > ymax:
            # move P1
            P1.set_data(xmin + x_shift, ymin + y_shift)
            reinit = True
        if x_P2 < xmin or x_P2 > xmax or y_P2 < ymin or y_P2 > ymax:
            # move P2
            P2.set_data(xmax - x_shift, ymax - y_shift)
            reinit = True
        if reinit:
            # redraw the line between P1 and P2, and recompute the growth
            dpoints.draw_lines()
            compute_growth(P1, P2, plot_residuals=True)
        fig_growth_rate.canvas.draw()

    zone_selector = RectangleSelector(axSelect, on_select,
            drawtype='box', useblit=True, button=[1, 3],  # don't use middle button
            minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    def on_draw(evt):
        zone_selector.set_visible(True)

    fig_growth_rate.canvas.mpl_connect('draw_event', on_draw)

#     fig_growth_rate.tight_layout()
    grid.tight_layout(fig_growth_rate)
    fig_growth_rate.show()


def show_growth_length():
    """Interactive estimation of the growth length.

    The user can select the point on the organ from which the growth begins.
    """
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Retrieve the detection step from the h5 file
    detection_step = h5store.get_detection_step(hdf5file)

    if detection_step is None:
        detection_step = DETECTION_STEP_DEFAULT

    # Recup l'id de la tige dans le fichier h5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Scaling factor (conversion from pixels to cm)
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Récuperation des données dont on a besoin
    d, d, s, d, angles, d, d, d = load_postprocessed_data(hdf5file, hdf_tige_id)
    # 'd' stands for 'dummy'

    # Scaling
    length_unit, inv_length_unit = "pix", r"pix$^{-1}$"
    if scale_cmpix is not None:
        length_unit, inv_length_unit = "cm", r"cm$^{-1}$"
        s *= scale_cmpix

    image = cur_image
    initial_image = 0

    times, time_units, time_label, graduation, excluded_image, mask = get_time_data()

    xc, yc = tiges_data.xc[cur_tige].copy(), tiges_data.yc[cur_tige].copy()
    # Removing shifts along the y-axis due to incorrect camera angles
    yc0_ref = yc[initial_image, 0]
    yc -= yc[:, 0, newaxis] - yc0_ref

    # If the steady-state image has to be excluded from time series
    # (for instance because it has been taken a very long time after other images)
    if excluded_image is not None:
        times = times[mask]
        xc = xc[mask]
        yc = yc[mask]
        s = s[mask]
        angles = angles[mask]
        if excluded_image == cur_image:
            # If the current image is the steady-state image to exclude,
            # just take the previous image as the current image.
            image = excluded_image - 1

    time = times[image]
    time_unit = time_units["time unit"]
    if time_unit == strings["image"]:
        line_label = strings["image"] + " %i"%image  # image number
    else:  # if there is a proper time unit
        line_label = "t = %.1f "%time + time_unit

    xmin, xmax = xc.min(), xc.max()
    ymin, ymax = yc.min(), yc.max()

    imgxmin = int(xmin - 0.02*xmin)
    imgxmax = int(xmax + 0.02*xmax)
    imgymin = int(ymin - 0.02*ymin)
    imgymax = int(ymax + 0.02*ymax)

    # Creating the figure
    start_title = _("Estimation of the growth length for organ")
    fig_growth_length = mpl.figure(start_title + " %i"%(cur_tige+1), figsize=(12, 10))

#     grid = mpl.GridSpec(2, 2, wspace=.2, hspace=.2)
    grid = mpl.GridSpec(2, 2)

    axProfiles = fig_growth_length.add_subplot(grid[0, 1])

    t0 = h5store.open_hdf5_image(hdf5file, initial_image, 0)
    img0 = t0[imgymin:imgymax, imgxmin:imgxmax]
    axProfiles.imshow(img0, 'gray', alpha=0.7)

    axProfiles.plot(xc[0] - imgxmin, yc[0] - imgymin, 'g-', lw=2)
    pl_profile_cur, = axProfiles.plot(xc[image] - imgxmin,
                                      yc[image] - imgymin, 'm--', lw=2)
    axProfiles.axis('equal')
    axProfiles.axis('off')

    # Size of the averaging window
    W = int(4 * round(tiges_data.diam[cur_tige].mean() / detection_step))

    # Smoothing and computation of the curvatures
    smooth_angles = zeros_like(angles) - 3000
    smooth_s = zeros_like(angles) - 3000
    curvatures = zeros_like(angles) - 3000
    cutoff_left, cutoff_right = 2, 2
    for i in range(len(angles)):
        imax = flatnonzero(angles[i].mask == False).max() + 1
        angles_i = angles[i, cutoff_left:imax-cutoff_right]
        s_i = s[i, cutoff_left:imax-cutoff_right]
        curvatures_i, smooth_angles_i = get_tige_curvature(angles_i, s_i, smoothing=True,
                                                           window_width=W)
        smooth_angles_i -= smooth_angles_i[0]
        smooth_angles_i = rad2deg(smooth_angles_i)
        if interekt.angle_0_360.get():
            smooth_angles_i = convert_angle(smooth_angles_i)
        smooth_angles[i, :len(smooth_angles_i)] = smooth_angles_i
        smooth_s[i, :len(s_i)] = s_i
        curvatures[i, :len(curvatures_i)] = curvatures_i

    smooth_angles = ma.masked_equal(smooth_angles, -3000)
    smooth_angles = ma.masked_invalid(smooth_angles)
    smooth_s = ma.masked_equal(smooth_s, -3000)
    smooth_s = ma.masked_invalid(smooth_s)
    curvatures = ma.masked_equal(curvatures, -3000)
    curvatures = ma.masked_invalid(curvatures)

    t_min, t_max = times.min(), times.max()
    s_max = smooth_s.max()
    angle_min, angle_max = smooth_angles.min(), smooth_angles.max()
    curvature_min, curvature_max = curvatures.min(), curvatures.max()

    axAngle = fig_growth_length.add_subplot(grid[0,0])
    axCurvatureMap = fig_growth_length.add_subplot(grid[1,:])

    pl_angles_cur, = axAngle.plot(smooth_s[image],
                                  smooth_angles[image],
                                  'm', label=line_label)
    axAngle.plot(smooth_s[initial_image],
                 smooth_angles[initial_image],
                 'g', label=strings["initial"])
    axAngle.set_xlim([-s_max/50, s_max + s_max/50])
    axAngle.set_ylim([angle_min, angle_max])
    axAngle.set_xlabel('s (%s)'%length_unit)
    axAngle.set_ylabel(strings["angle"] + r" $A - A(s=0)$ " + strings["degree"])
    axAngle.grid(True)
    axAngle.legend()

    imax = flatnonzero(smooth_s[-1].mask == False).max() + 1
    curvatures_resized = curvatures[:,:imax]
    cmap = mpl.get_cmap('jet')
    colorm = axCurvatureMap.pcolormesh(smooth_s[-1, :imax], times,
                                       curvatures_resized,
                                       norm=SymLogNorm(linthresh=0.6, linscale=5),
                                       # linthresh highest normal curvature observed
                                       cmap=cmap, rasterized=True)
    cbar = mpl.colorbar(colorm, use_gridspec=True, pad=0.01, extend='both')
    cbar.ax.tick_params(labelsize=8)
    axCurvatureMap.set_xlim([-s_max/100, s_max + s_max/100])
    axCurvatureMap.set_ylim([-t_max/50, t_max + t_max/50])
    axCurvatureMap.set_xlabel(strings["s"] + ' (%s)'%length_unit)
    axCurvatureMap.set_ylabel(time_label)

    # The growth length will be stored in growth_data.
    growth_data = h5store.get_postprocessing(hdf5file, 'growth_data', hdf_tige_id)

    # The growth rate should have been already estimated and stored in growth_data.
    try:
        dLdt = growth_data["dLdt"]
    except:
        msg = _("The growth rate has to be estimated before the growth length.")
        messagebox.showerror(strings["Missing data"], msg)
        mpl.close(fig_growth_length)
        return

    # array index for the beginning of the growth zone at initial time
    # In case it has already been estimated.
    if 'i_gz0' in growth_data:
        i_gz0 = growth_data['i_gz0']
    else:
        i_gz0 = 0

    # Check if the value of i_gz0 is possible, otherwise set it at a bound
    if i_gz0 < 0:
        i_gz0 = 0
    elif i_gz0 >= len(smooth_s[initial_image]):
        i_gz0 = len(smooth_s[0]) - 1

    # arc-length at the beginning of the growth zone at initial time, s_gz0
    s_gz0 = smooth_s[initial_image, i_gz0]

    t1, t2, L1 = growth_data["t1"], growth_data["t2"], growth_data["L1"]
    if time_unit == strings["min"]:
        # The growth rate (dL/dt) is in [L]h¯¹.
        # Hence the division by 60 to come back to [L]min¯¹.
        lengths_fitted = dLdt/60 * (times - t1) + L1
    elif time_unit == strings["days"]:
        # The growth rate (dL/dt) is in [L]h¯¹.
        # Hence the division by 60 to come back to [L]day¯¹.
        lengths_fitted = dLdt * 24 * (times - t1) + L1
    else:
        lengths_fitted = dLdt * (times - t1) + L1
    axCurvatureMap.plot(lengths_fitted, times, 'k--')
    try:
        i1 = flatnonzero(times >= t1)[0]
    except:
        ii = 0
    try:
        i2 = flatnonzero(times >= t2)[0]
    except:
        i2 = -1
    axCurvatureMap.plot(lengths_fitted[i1:i2], times[i1:i2], 'k-')

    # growth increments from initial length
    growth_increments = lengths_fitted - lengths_fitted[0]

    # arc-length at the beginning of the growth zone (for all times)
    s_gz = growth_increments + s_gz0

    pl_s_curvaturemap, = axCurvatureMap.plot([s_gz0, s_gz[-1]],
                                             axCurvatureMap.get_ylim(),
                                             'r--', lw=3, picker=10)

    pl_t_curvaturemap, = axCurvatureMap.plot(axCurvatureMap.get_xlim(), [time]*2,
                                             'm--', lw=3, picker=10)

    dlines = DraggableLines([pl_s_curvaturemap, pl_t_curvaturemap], directions=["h", "v"])

    # big red dot on the tige image at s_gz0
    pl_s_gz0, = axProfiles.plot(xc[initial_image, i_gz0] - imgxmin,
                                yc[initial_image, i_gz0] - imgymin, 'ro', ms=8)

    # big red dot on the on the angle plot at s_gz0
    pl_s_angle_gz0, = axAngle.plot(s_gz0, smooth_angles[initial_image, i_gz0],
                                   'ro', ms=8)

    # big red dots at s_gz at current time
    s_gz_cur = s_gz[image]
    i_gz_cur = flatnonzero(smooth_s[image] >= s_gz_cur)[0]
    pl_s_gz_cur, = axProfiles.plot(xc[image, i_gz_cur] - imgxmin,
                                   yc[image, i_gz_cur] - imgymin, 'ro', ms=8)
    pl_s_angle_gz_cur, = axAngle.plot(s_gz_cur, smooth_angles[image, i_gz0],
                                   'ro', ms=8)

    # growth length
    Lgz = lengths_fitted[0] - s_gz0

    text_infos = mpl.figtext(
            .5, 0.01,
            strings["growth length"] + " (Lgz): %0.2f%s"%(Lgz, length_unit),
            fontsize=11, ha='center', color='Red')

    def on_pick(evt):
        dlines.on_press(evt)

    def on_motion(evt):
        dlines.on_motion(evt)

    def on_release(evt):
        dlines.on_release(evt)

        if dlines.changed:
            dlines.changed = False

            time = pl_t_curvaturemap.get_data()[1][0]
            if time < t_min:
                time = t_min
            elif time > t_max:
                time = t_max
            image = flatnonzero(times >= time)[0]
            time = times[image]
            if time_unit == strings["image"]:
                line_label = strings["image"] + " %i"%image  # image number
            else:  # if there is a proper time unit
                line_label = "t = %.1f "%time + time_unit

            # arc-length at the beginning of the growth zone
            s_gz0 = pl_s_curvaturemap.get_data()[0][0]
            if s_gz0 < 0:
                s_gz0 = 0
            elif s_gz0 > smooth_s[initial_image].max():
                s_gz0 = smooth_s[initial_image].max()

            # arc-length at the beginning of the growth zone (for all times)
            s_gz = growth_increments + s_gz0

            # array index at the beginning of the growth zone
            i_gz0 = flatnonzero(smooth_s[initial_image] >= s_gz0)[0]

            s_gz_cur = s_gz[image]
            i_gz_cur = flatnonzero(smooth_s[image] >= s_gz_cur)[0]

            pl_profile_cur.set_data(xc[image] - imgxmin, yc[image] - imgymin)

            pl_s_gz0.set_data(xc[0, i_gz0] - imgxmin,
                              yc[0, i_gz0] - imgymin)
            pl_s_gz_cur.set_data(xc[image, i_gz_cur] - imgxmin,
                                 yc[image, i_gz_cur] - imgymin)

            pl_angles_cur.set_data(smooth_s[image], smooth_angles[image])
            pl_angles_cur.set_label(line_label)
            axAngle.legend(loc=0, prop={'size':10})

            pl_s_angle_gz0.set_data(s_gz0, smooth_angles[initial_image, i_gz0])
            pl_s_angle_gz_cur.set_data(s_gz_cur, smooth_angles[image, i_gz0])

            # growth length
            Lgz = lengths_fitted[0] - s_gz0

            # Relative Elongation Rate
            RER = dLdt / Lgz

            growth_data["i_gz0"] = i_gz0
            growth_data["s_gz0"] = s_gz0
            growth_data["Lgz"] = Lgz
            growth_data["RER"] = RER

            # Saving growth_data in the hdf5 file
            h5store.save_tige_dict_to_hdf(
                    hdf5file, hdf_tige_id, {'growth_data': growth_data})

            text_infos.set_text(
                    strings["growth length"] + " (Lgz): %0.2f%s"%(Lgz, length_unit))

            fig_growth_length.canvas.draw()

    def on_close(evt):
        mpl.close(fig_growth_length)

    fig_growth_length.canvas.mpl_connect('pick_event', on_pick)
    fig_growth_length.canvas.mpl_connect('button_release_event', on_release)
    fig_growth_length.canvas.mpl_connect('motion_notify_event', on_motion)
    fig_growth_length.canvas.mpl_connect('close_event', on_close)

#     fig_growth_length.tight_layout()
    # The '0.05' in the 'rect' keyword argument ensures that 'text_infos' does not
    # overlap with an axis label at the bottom of the figure.
    grid.tight_layout(fig_growth_length, rect=[0, 0.05, 1, 1])
    fig_growth_length.show()


def show_beta_tilde():
    r"""Interactive estimation of the dimensionless graviceptive sensitivity β̃.

    The definition of β̃ ($\tilde{\beta}$) can be found in :
    Bastien, Douady and Moulia (2014), https://doi.org/10.3389/fpls.2014.00136.
    """
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Retrieving the tige id from the hdf5 file
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)
    # Tige name
    tige_name = h5store.get_tiges_names(hdf5file, hdf_tige_id)
    # Retrieving scaling data from the hdf5 file
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Retrieving useful data
    radius = tiges_data.diam[cur_tige].mean() / 2.   # tige radius
    x, y, s, lengths, angles, tip_angles, dummy, lines =\
            load_postprocessed_data(hdf5file, hdf_tige_id)

    # Scaling
    length_unit = "pix"
    if scale_cmpix is not None:
        length_unit = "cm"
        radius *= scale_cmpix
        x *= scale_cmpix
        y *= scale_cmpix
        s *= scale_cmpix
        lengths *= scale_cmpix

    times, time_units, time_label, graduation, excluded_image, mask = get_time_data()

    if len(times) <= 1:
        return

    # If the steady-state image has to be excluded from time series
    # (for instance because it has been taken a very long time after other images)
    if excluded_image is not None:
        times = times[mask]
        s = s[mask]
        lines = lines[mask]
        lengths = lengths[mask]
        angles = angles[mask]
        tip_angles = tip_angles[mask]

    # create unit strings
    time_unit = time_units["time unit"]
    dAdt_unit = time_units["dAdt unit"]
    dAdt_unit_TeX = time_units["dAdt unit TeX"]

    start_title = _("Estimation of beta for organ")
    fig_beta = mpl.figure(start_title + ' %s'%str(tige_name), figsize=(10, 8))

#     grid = mpl.GridSpec(2, 3, wspace=.4, hspace=.4)
    grid = mpl.GridSpec(2, 3)

    # Displaying tige profiles
    axProfiles = fig_beta.add_subplot(grid[:,0])
    lcollec = LineCollection(lines, linewidth=(2,), cmap='viridis')
    lcollec.set_array(times)
    axProfiles.add_collection(lcollec)
    axProfiles.set_xlim((x.min(), x.max()))
    axProfiles.set_ylim((y.min(), y.max()))
    axProfiles.set_xlabel("x - x0 (%s)"%length_unit)
    axProfiles.set_ylabel("y - y0 (%s)"%length_unit)
    axProfiles.axis('equal')

    axAngle = fig_beta.add_subplot(grid[0, 1:])
    axLength = fig_beta.add_subplot(grid[1, 1:], sharex=axAngle)

    xlims = (times[0], times[-1])
    axAngle.set_xlim(xlims)
    axAngle.grid(True)
    axAngle.xaxis.set_major_locator(MultipleLocator(graduation))
    axLength.grid(True)

    # Les tailles min et max sont resp. le min/max sur les 5 premiers/derniers points.
    length_min = min(lengths[:5])
    length_max = max(lengths[-5:])

    # Tracé de la taille en fonction du temps
    length_mean = (length_min + length_max) / 2
    span = 300    # ask Jérôme for details
    if scale_cmpix is not None:
        span *= scale_cmpix
    axLength.plot(times, lengths, '+-', color=tiges_colors[cur_tige], lw=2)
    axLength.set_ylim((length_mean - span/2, length_mean + span/2))
    axLength.set_xlabel(time_label)
    axLength.set_ylabel(strings["length"] + " (%s)"%length_unit)

    # The growth rate should have been alreedy estimated and stored in growth_data.
    growth_data = h5store.get_postprocessing(hdf5file, 'growth_data', hdf_tige_id)
    try:
        dLdt = growth_data["dLdt"]
    except:
        msg = _("The growth rate has to be estimated before beta.")
        messagebox.showerror(strings["Missing data"], msg)
        mpl.close(fig_beta)
        return

    t1, t2, L1 = growth_data["t1"], growth_data["t2"], growth_data["L1"]
    if time_unit == strings["min"]:
        # The growth rate (dL/dt) is in [L]h¯¹.
        # Hence the division by 60 to come back to [L]min¯¹.
        lengths_fitted = dLdt/60 * (times - t1) + L1
    elif time_unit == strings["days"]:
        # The growth rate (dL/dt) is in [L]h¯¹.
        # Hence the division by 60 to come back to [L]day¯¹.
        lengths_fitted = dLdt * 24 * (times - t1) + L1
    else:
        lengths_fitted = dLdt * (times - t1) + L1
    axLength.plot(times, lengths_fitted, 'k--')
    try:
        i1 = flatnonzero(times >= t1)[0]
    except:
        ii = 0
    try:
        i2 = flatnonzero(times >= t2)[0]
    except:
        i2 = -1
    axLength.plot(times[i1:i2], lengths_fitted[i1:i2], 'k-')

    if interekt.angle_0_360.get():
        tip_angles = convert_angle(tip_angles)

    axAngle.plot(times, tip_angles, '+-', color=tiges_colors[cur_tige], lw=2)
    axAngle.set_ylabel(strings["tip angle"] + " " + strings["degree"])
    axAngle.set_xlabel(time_label)

    # Taquets mobiles pour définir la plage de fit
    try:
        select_start, = axAngle.plot([times[5]] * 2, axAngle.get_ylim(),
                                     'b', lw=2, picker=5)
        select_end, = axAngle.plot([times[-5]] * 2, axAngle.get_ylim(),
                                   'b', lw=2, picker=5)
    except:
        print('Not enough images for estimation of beta.')
        select_start, = axAngle.plot([times[0]] * 2, axAngle.get_ylim(), 'b',
                                     lw=2, picker=5)
        select_end, = axAngle.plot([times[0]] * 2, axAngle.get_ylim(), 'b',
                                   lw=2, picker=5)

    plfit_angles, = axAngle.plot([], [], 'k', lw=1)
    dlines = DraggableLines([select_start, select_end])
    xmin, xmax = axProfiles.get_xlim()
    text_beta = axProfiles.text(xmin + (xmax - xmin) / 10,
                                axProfiles.get_ylim()[-1], '')
    text_R2 = axAngle.text(.9, .9, '', ha='right', va='top', transform=axAngle.transAxes)

    def fit_beta():
        global beta_data   # is it necessary?

        ta = select_start.get_xdata()[0]
        tb = select_end.get_xdata()[0]
        # Reorder the boundaries
        t1 = min(ta, tb)
        t2 = max(ta, tb)
        # Find indices
        i1 = flatnonzero(times >= t1)[0]
        i2 = flatnonzero(times >= t2)[0]

        fit_zone = slice(i1, i2)
        fit_tip_angles, res, _, _, _ = ma.polyfit(times[fit_zone], tip_angles[fit_zone],
                                                  1, full=True)
        dAdt, intercept = fit_tip_angles[0], fit_tip_angles[1]
        plfit_angles.set_data(times[fit_zone], dAdt * times[fit_zone] + intercept)

        S_res = res[0]
        S_tot = ma.sum((tip_angles[fit_zone] - mean(tip_angles[fit_zone]))**2)
        R2 = 1 - S_res / S_tot

        dLdt = growth_data["dLdt"]
        dLdt_unit_TeX = growth_data["dLdt_unit_TeX"].astype(str)

        dAdt = abs(deg2rad(dAdt))
        # If photo timestamps are given, the Relative Elongation Rate (RER) is in h¯¹.
        # We want dAdt in the same unit.
        if time_unit == strings["min"]:
            dAdt *= 60
        elif time_unit == strings["days"]:
            dAdt /= 24

        angle_init = abs(angles[0].mean()) #Angle moyen de la tige au temps 0 (deja en radians)
        beta_tilde = radius / sin(angle_init) * (dAdt / dLdt)

        text_beta.set_text(r"""$\tilde{\beta} = %0.2f$"""
                           """\n"""
                           r"""$R = %0.2f$ %s"""
                           """\n"""
                           r"""$A_{init} = %0.1f^{\circ}$"""
                           """\n"""
                           r"""$\frac{dA}{dt} = %0.2f$ %s"""
                           """\n"""
                           r"""$\frac{dL}{dt} = %0.4f$ %s"""
                           """\n""" %(beta_tilde, radius, length_unit, angle_init*180/pi,
                                      dAdt, dAdt_unit_TeX, dLdt, dLdt_unit_TeX))

        text_R2.set_text(r"$R^2 = %0.3f$"%R2)

        fig_beta.canvas.draw_idle()

        beta_data = {"t1":t1, "t2":t2, 'i1':i1, 'i2':i2, 'beta_tilde':beta_tilde,
                     "radius":radius, "Ainit":angle_init, "dAdt":dAdt, 'R2':R2,
                     "dAdt_unit":dAdt_unit}

        # Sauvegarde des données dans le fichier hdf5
        h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id, {'beta_data': beta_data})

    def init_fit():
        # Retrieving beta_data in the hdfR file
        tigeid = get_hdf5_tigeid(hdf5file, cur_tige)
        beta_data = h5store.get_postprocessing(hdf5file, 'beta_data', tigeid)
        if beta_data is not None:
            t1, t2 = beta_data['t1'], beta_data['t2']
            if "Ainit(rad)" in beta_data:
                # Data from outdated version, should be deleted.
                h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id, {'beta_data':None})
        else:
            # Automatic setting of the boundaries for the fit
            i2 = tip_angles.argmax() - 2
            i1 = tip_angles[:i2].argmin()
            t1, t2 = times[i1], times[i2]
        try:
            select_start.set_xdata((t1, t1))
            select_end.set_xdata((t2, t2))
            fit_beta()
        except:
            print("Erreur dans l'initiation de l'ajustement.")
            print(t1, t2)
            # Automatic setting of the boundaries for the fit
            i2 = tip_angles.argmax() - 2
            i1 = tip_angles[:i2].argmin()
            t1, t2 = times[i1], times[i2]
            select_start.set_xdata((t1, t1))
            select_end.set_xdata((t2, t2))
            fit_beta()

    init_fit()

    def on_pick(e):
        dlines.on_press(e)

    def on_motion(e):
        dlines.on_motion(e)

    def on_release(e):
        dlines.on_release(e)
        if axAngle.contains(e)[0]:
            fit_beta()

    def on_close(event):
        mpl.close(fig_beta)   # faut-il changer ça ?

    # Add click event for the figure
    fig_beta.canvas.mpl_connect('close_event', on_close)
    fig_beta.canvas.mpl_connect('pick_event', on_pick)
    fig_beta.canvas.mpl_connect('button_release_event', on_release)
    fig_beta.canvas.mpl_connect('motion_notify_event', on_motion)

#     fig_beta.tight_layout()
    grid.tight_layout(fig_beta)

    # Show the figure
    fig_beta.show()


def show_gamma_tilde():
    r"""Interactive estimation of the dimensionless proprioceptive sensitivity γ̃.

    The definition of γ̃ ($\tilde{\gamma}$) can be found in :
    Bastien, Douady and Moulia (2014), https://doi.org/10.3389/fpls.2014.00136.

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
    """
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Retrieving the tige id from the hdf5 file
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)
    # Tige name
    tige_name = h5store.get_tiges_names(hdf5file, hdf_tige_id)
    # Retrieving scaling data from the hdf5 file
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Retrieving useful data
    x, y, s, lengths, angles, tip_angles, d, lines =\
            load_postprocessed_data(hdf5file, hdf_tige_id)

    # Scaling
    length_unit = "pix"
    if scale_cmpix is not None:
        length_unit = "cm"
        x *= scale_cmpix
        y *= scale_cmpix
        s *= scale_cmpix
        lengths *= scale_cmpix

    times, time_units, time_label, graduation, excluded_image, mask = get_time_data()

    # If the steady-state image has to be excluded from time series
    # (for instance because it has been taken a very long time after other images)
    if excluded_image is not None:
        times = times[mask]
        s = s[mask]
        lines = lines[mask]
        lengths = lengths[mask]
        angles = angles[mask]
        tip_angles = tip_angles[mask]

    if len(times) <= 1:
        return

    # create unit strings
    time_unit = time_units["time unit"]
    gamma_unit = time_units["RER unit"]
    gamma_unit_TeX = time_units["RER unit TeX"]

    if time_unit == strings["image"]:
        messagebox.showerror(
                _("""No times associated with photographs"""),
                _("""Without time data, the computation of gamma will be meaningless. """
"""The computation of gamma_tilde is still possible."""))

    start_title = _("Estimation of gamma for organ")
    fig_gamma = mpl.figure(start_title + ' %s'%str(tige_name), figsize=(10, 8))

#     grid = mpl.GridSpec(6, 3, wspace=.4, hspace=.4)
    grid = mpl.GridSpec(6, 3)

    # Displaying tige profiles
    axProfiles = fig_gamma.add_subplot(grid[:,0])
    lcollec = LineCollection(lines, linewidth=(2,), cmap='viridis')
    lcollec.set_array(times)
    axProfiles.add_collection(lcollec)
    axProfiles.set_xlim((x.min(), x.max()))
    axProfiles.set_ylim((y.min(), y.max()))
    axProfiles.set_xlabel("x - x0 (%s)"%length_unit)
    axProfiles.set_ylabel("y - y0 (%s)"%length_unit)
    axProfiles.axis('equal')

    # Displaying the zones on which the mean angle is computed
    for t in [0]:
        xt = x[t, ~x[t].mask]
        yt = y[t, ~y[t].mask]
        try:
            xlims = [xt[tige_measure_zone[t][0]],
                     xt[tige_measure_zone[t][1]]]
            ylims = [yt[tige_measure_zone[t][0]],
                     yt[tige_measure_zone[t][1]]]
            colortige, = axProfiles.plot(xt, yt, 'k', lw=2.5)
            lims, = axProfiles.plot(xlims, ylims, 'o', color='m', ms=10)
        except:
            pass

    axLogAngle = fig_gamma.add_subplot(grid[0:2, 1:])
    axAngle = fig_gamma.add_subplot(grid[2:4, 1:], sharex=axLogAngle)
    axLength = fig_gamma.add_subplot(grid[4:, 1:], sharex=axLogAngle)

    xlims = (times[0], times[-1])
    axLogAngle.set_xlim(xlims)
    axLogAngle.grid(True)
    axLogAngle.xaxis.set_major_locator(MultipleLocator(graduation))
    axAngle.grid(True)
    axAngle.set_ylim((-30, 100))
    axLength.grid(True)

    # Les tailles min et max sont resp. le min/max sur les 5 premiers/derniers points.
    length_min = min(lengths[:5])
    length_max = max(lengths[-5:])

    # Tracé de la taille en fonction du temps
    length_mean = (length_min + length_max) / 2
    span = 300    # ask Jérôme for details
    if length_unit == "cm":
        span *= scale_cmpix
    axLength.plot(times, lengths, '+-', color=tiges_colors[cur_tige], lw=2)
    axLength.set_ylim((length_mean - span/2, length_mean + span/2))
    axLength.set_xlabel(time_label)
    axLength.set_ylabel(strings["length"] + " (%s)"%length_unit)

    # The Relative Elongation Rate (RER) should have been alreedy estimated
    # through the estimation of the growth rate and the growth length.
    growth_data = h5store.get_postprocessing(hdf5file, 'growth_data', hdf_tige_id)
    try:
        RER = growth_data["RER"]
    except:
        msg = _("The growth length has to be estimated before gamma.")
        messagebox.showerror(strings["Missing data"], msg)
        mpl.close(fig_gamma)
        return

    # Check whether time units are consistent
    if time_units["RER unit"] != growth_data["RER_unit"].astype(str):
        msg = _("""Time units do not match. """
"""Compute the growth rate and growth length again.""")
        messagebox.showerror(strings["Unit mismatch"], msg)
        return

    # Plot linearized growth
    dLdt = growth_data["dLdt"]
    t1, t2, L1 = growth_data["t1"], growth_data["t2"], growth_data["L1"]
    if time_unit == strings["min"]:
        # The growth rate (dL/dt) is in [L]h¯¹.
        # Hence the division by 60 to come back to [L]min¯¹.
        lengths_fitted = dLdt/60 * (times - t1) + L1
    elif time_unit == strings["days"]:
        # The growth rate (dL/dt) is in [L]h¯¹.
        # Hence the division by 60 to come back to [L]day¯¹.
        lengths_fitted = dLdt * 24 * (times - t1) + L1
    else:
        lengths_fitted = dLdt * (times - t1) + L1
    axLength.plot(times, lengths_fitted, 'k--')
    i1 = flatnonzero(times >= t1)[0]
    i2 = flatnonzero(times >= t2)[0]
    axLength.plot(times[i1:i2], lengths_fitted[i1:i2], 'k-')

    angles = rad2deg(angles)

    if interekt.angle_0_360.get():
        angles = convert_angle(angles)
        tip_angles = convert_angle(tip_angles)

    # Il faut soustraire l'angle en début de tige A(s = 0)
    angle_mean = angles.mean(axis=0)   # moyenne temporelle
    angle_mean_s0 = ma.mean(angle_mean[:10])    # moyenne spatiale en début de tige
    angle_deviations = tip_angles - angle_mean_s0   # angle deviation from base to tip

    log_A_dev = log(angle_deviations)
    axLogAngle.plot(times, log_A_dev, '+-', color=tiges_colors[cur_tige], lw=2)
    axLogAngle.set_ylabel(r'$\log(A_{bout} - A_{base})$')
    axLogAngle.set_xlabel(time_label)

    axAngle.plot(times, angle_deviations, '+-', color=tiges_colors[cur_tige], lw=2)
    axAngle.set_ylabel(r'$A_{bout} - A_{base}$ (deg)')
    axAngle.set_xlabel(time_label)

    def fit_gamma():
        global gamma_data   # is it necessary?

        ta = select_start.get_xdata()[0]
        tb = select_end.get_xdata()[0]
        t1 = min(ta, tb)
        t2 = max(ta, tb)

        # Indices de début et fin de la plage de fit
        i1 = flatnonzero(times >= t1)[0]
        i2 = flatnonzero(times >= t2)[0]

        # Tableaux pour le fit de la différence d'angle
        time_range = times[i1:i2]
        A_dev_range = angle_deviations[i1:i2]
        log_A_dev_range = log_A_dev[i1:i2]

        # La stratégie consiste à passer au log pour se ramener à une régression
        # linéaire. Ce passage pose cependabt un problème du point de vue de la théorie
        # de la régression linéaire. Il y a un biais en faveur des petites valeurs de
        # angle_deviation. Le moyen propre d'y remédier est d'ajouter une pondération
        # sous la forme de l'argument 'w=ma.sqrt(A_ecart_plage)' dans l'appel à polyfit.
        # Sources :
        #   - http://mathworld.wolfram.com/LeastSquaresFittingExponential.html
        #   - https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
        weights = ma.sqrt(A_dev_range)
        fit_log_A_dev, res, _, _, _ = ma.polyfit(time_range, log_A_dev_range, 1,
                                                 w=weights, full=True)

        weighted_mean = mean(weights * log_A_dev_range)
        S_res, S_tot = res[0], ma.sum((weights * log_A_dev_range - weighted_mean)**2)
        R2 = 1 - S_res / S_tot

        slope, intercept = fit_log_A_dev[0], fit_log_A_dev[1]
        log_A_dev_fitted = slope * times + intercept
        plfit_log_A_dev.set_data(time_range, log_A_dev_fitted[i1:i2])
        plfit_A_dev.set_data(times[i1:], exp(log_A_dev_fitted[i1:]))

        # gamma is the opposite of the slope of log_A(t):
        gamma = -slope
        # If photo timestamps are given, we want gamma in the same unit as
        # the Relative Elongation Rate (RER), which is in h¯¹.
        if time_unit == strings["min"]:
            gamma *= 60
        elif time_unit == strings["days"]:
            gamma /= 24

        gamma_tilde = gamma / RER

        RER_unit_TeX = growth_data["RER_unit_TeX"].astype(str)
        text = (r"""$\gamma = %0.3f$ %s"""
                """\n"""
                r"""$\tilde{\gamma} = %0.2f$"""
                """\n"""
                r"""$\dot{E} = %0.4f$ %s"""
                """\n"""%(gamma, gamma_unit_TeX, gamma_tilde, RER, RER_unit_TeX))

        text_gamma.set_text(text)
        text_R2.set_text(r"$R^2 = %0.3f$"%R2)

        fig_gamma.canvas.draw_idle()

        gamma_data = {"t1":t1, "t2":t2, 'i1':i1, 'i2':i2,
                      "gamma":gamma, 'R2':R2, "gamma_unit":gamma_unit,
                      "gamma_tilde": gamma_tilde}

        # Saving gamma_data in the hdf5 file
        h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id, {'gamma_data': gamma_data})

    def init_fit():
        # Récupération de gamma_data dans le fichier hdf5
        tigeid = get_hdf5_tigeid(hdf5file, cur_tige)
        gamma_data = h5store.get_postprocessing(hdf5file, 'gamma_data', tigeid)
        if gamma_data is not None:
            if "ideb" in gamma_data:
                t1, t2 = times[gamma_data['ideb']], times[gamma_data['ifin']]
                # Data from outdated version, should be deleted.
                h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id, {'gamma_data':None})
            else:
                t1, t2 = gamma_data['t1'], gamma_data['t2']
        else:
            # Automatic setting of the boundaries for the fit
            try:
                i1 = angle_deviations.argmax() + 2
                if any(angle_deviations[i1:] <= 0):
                    i2 = i1 + nonzero(angle_deviations[i1:] <= 0)[0][0] - 3
                else:
                    i2 = -5
                t1, t2 = times[i1], times[i2]
            except:
                print("""Échec de l'ajustement automatique.\nSélectionnez les bornes
                         manuellement.""")
                t1, t2 = times[0], times[-1]
        try:
            select_start.set_xdata((t1, t1))
            select_end.set_xdata((t2, t2))
            fit_gamma()
        except:
            print("Erreur dans l'initiation de l'ajustement.")
            print(t1, t2)

    # Taquets mobiles pour définir la plage de fit
    try:
        select_start, = axLogAngle.plot([times[5]] * 2, axLogAngle.get_ylim(), 'b',
                                         lw=2, picker=5)
        select_end, = axLogAngle.plot([times[-5]] * 2, axLogAngle.get_ylim(), 'b',
                                       lw=2, picker=5)
    except:
        print('Not enough images for estimation of gamma.')
        select_start, = axLogAngle.plot([times[0]] * 2, axLogAngle.get_ylim(), 'b',
                                        lw=2, picker=5)
        select_end, = axLogAngle.plot([times[0]] * 2, axLogAngle.get_ylim(), 'b',
                                      lw=2, picker=5)

    plfit_A_dev, = axAngle.plot([], [], 'k', lw=1)
    plfit_log_A_dev, = axLogAngle.plot([], [], 'k', lw=1)
    dlines = DraggableLines([select_start, select_end])
    xmin, xmax = axProfiles.get_xlim()
    text_gamma = axProfiles.text(xmin + (xmax - xmin) / 10,
                                 axProfiles.get_ylim()[-1], '')
    text_R2 = axLogAngle.text(0.9, 0.9, '',
                              ha='right', va='top', transform=axLogAngle.transAxes)
    init_fit()

    def on_pick(e):
        dlines.on_press(e)

    def on_motion(e):
        dlines.on_motion(e)

    def on_release(e):
        dlines.on_release(e)
        if axLogAngle.contains(e)[0]:
            # Ajustement pour trouver gamma puis calcul du RER pour obtenir gamma_tilde
            fit_gamma()

    def on_close(event):
        mpl.close(fig_gamma)   # faut-il changer ça ?

    #Add click event for the figure
    fig_gamma.canvas.mpl_connect('close_event', on_close)
    fig_gamma.canvas.mpl_connect('pick_event', on_pick)
    fig_gamma.canvas.mpl_connect('button_release_event', on_release)
    fig_gamma.canvas.mpl_connect('motion_notify_event', on_motion)

#     fig_gamma.tight_layout()
    grid.tight_layout(fig_gamma)

    #Show the figure
    fig_gamma.show()


def show_B(tige_id=None):
    """
    Estimate the balance number B as defined by the AC model of
    Bastien et al. 2013.
    """
    # Force la fermeture du menu popup dans tk
    interekt.floatmenuisopen = False

    # Retrieve the detection step from the h5 file
    detection_step = h5store.get_detection_step(hdf5file)

    if detection_step is None:
        detection_step = DETECTION_STEP_DEFAULT

    # Recup l'id de la tige dans le fichier h5
    hdf_tige_id = get_hdf5_tigeid(hdf5file, cur_tige)

    # Data will be stored in B_data
    B_data = h5store.get_postprocessing(hdf5file, 'B_data', hdf_tige_id)

    ss_image = interekt.steady_state_image
    if ss_image is None:
        if B_data is not None and 'num_img_fit' in B_data:
            # maybe the steady-state image has been selected in a previous analysis
            ss_image = B_data['num_img_fit']
        else:
            messagebox.showerror(
                    _("No steady-state image selected"),
                    _("Please select first the steady-state image."))
            return

    # Scaling factor (conversion from pixels to cm)
    scale_cmpix = h5store.get_pixelscale(hdf5file)

    # Récuperation des données dont on a besoin
    d, d, s, d, angles, d, d, d = load_postprocessed_data(hdf5file, hdf_tige_id)
    # 'd' is for 'dummy'

    # We just need the steady state
    s = s[ss_image]
    angles = angles[ss_image]

    # Scaling
    length_unit, inv_length_unit = "pix", r"pix$^{-1}$"
    if scale_cmpix is not None:
        length_unit, inv_length_unit = "cm", r"cm$^{-1}$"
        s *= scale_cmpix

    # Remove non-valid values
    imax = flatnonzero(angles.mask == False).max() + 1
    angles = angles[:imax]
    s = s[:imax]

    # Size of the averaging window
    W = int(4 * round(tiges_data.diam[cur_tige].mean() / detection_step))

    dummy, smooth_angles = get_tige_curvature(angles, s, smoothing=True, window_width=W)

    smooth_angles_deg = rad2deg(smooth_angles)
    if interekt.angle_0_360.get():
        smooth_angles_deg = convert_angle(smooth_angles)

    # The growth length is stored in growth_data
    growth_data = h5store.get_postprocessing(hdf5file, 'growth_data', hdf_tige_id)
    try:
        Lgz = growth_data["Lgz"]
        i_gz = growth_data["i_gz0"]
    except:
        messagebox.showerror(
                strings["Missing data"],
                _("The growth length has to be estimated before B."))
        return

    # Check if the value of i_gz is possible, otherwise set it at a bound
    if i_gz < 0:
        i_gz = 0
    elif i_gz >= len(s):
        i_gz = len(s) - 1

    # keep the growth zone only
    s_gz = s[i_gz:] - s[i_gz]
    angles_gz = angles[i_gz:]
    smooth_angles_gz = smooth_angles[i_gz:]
    smooth_angles_deg_gz = smooth_angles_deg[i_gz:]

    start_title = _("B for organ")
    fig_B = mpl.figure(start_title + ' %i'%(cur_tige), figsize=(12,10))

    grid = mpl.GridSpec(2,2)

    axProfiles = fig_B.add_subplot(grid[0,1])

    # Need organ skeleton at initial time
    ti_xc, ti_yc = tiges_data.xc[cur_tige, 0], tiges_data.yc[cur_tige, 0]

    # Need organ skeleton at final time
    tf_xc, tf_yc = tiges_data.xc[cur_tige, -1], tiges_data.yc[cur_tige, -1]

    xmax = max((ti_xc.max(), tf_xc.max()))
    xmin = min((ti_xc.min(), tf_xc.min()))
    ymax = max((ti_yc.max(), tf_yc.max()))
    ymin = min((ti_yc.min(), tf_yc.min()))

    imgxma = int(xmax + 0.02*xmax)
    imgxmi = int(xmin - 0.02*xmin)
    imgyma = int(ymax + 0.02*ymax)
    imgymi = int(ymin - 0.02*ymin)

    ss_photo = h5store.open_hdf5_image(hdf5file, ss_image, 0)
    ss_photo_cropped = ss_photo[imgymi:imgyma, imgxmi:imgxma]
    axProfiles.imshow(ss_photo_cropped, 'gray')

    # Plot the skeleton of the tige, from i_gz (i.e. only the growing zone)
    axProfiles.plot(tiges_data.xc[cur_tige, ss_image, i_gz:] - imgxmi,
                    tiges_data.yc[cur_tige, ss_image, i_gz:] - imgymi, 'r-', lw=2)

    axProfiles.axis('equal')
    axProfiles.axis('off')

    axAngle = fig_B.add_subplot(grid[0,0])
    axAngle.set_xlabel("s (%s)"%length_unit)
    axAngle.set_ylabel(strings["angle"] + " " + strings["degree"])
    axAngle.grid(True)

    axAngle.plot(s_gz, smooth_angles_deg_gz, 'g')

    angles_normalized = smooth_angles_gz / smooth_angles_gz[0]
    log_angles = ma.log(angles_normalized)

    #Fits sur A/A₀ et log(A/A₀)
    ax_Afit = fig_B.add_subplot(grid[1,:])
    ax_Afit.set_xlim(0, s_gz.max())
    ax_Afit.set_ylim(angles_normalized.min(), angles_normalized.max())
    ax_Afit.set_ylabel('A/A₀')
    ax_Afit.set_xlabel('s (%s)'%length_unit)
    ax_Afit.grid(True)

    pl_A, = ax_Afit.plot(s_gz, angles_normalized, 'o')
    #pl_Afit, = ax_Afit.plot([], [], 'r', lw=2, label='R')
    pl_logAfit_exp, = ax_Afit.plot([], [], 'r--', lw=2, label='R')

#    ax_logAfit = fig_B.add_subplot(grid[1,1])
#    ax_logAfit.set_ylabel("log(A/A₀)")
#    ax_logAfit.set_xlabel('s growth zone (%s)'%length_unit)
#    ax_logAfit.set_xlim(0, s_gz.max())
#    ax_logAfit.set_ylim(log_angles.min(), log_angles.max())
#    ax_logAfit.grid(True)

#    pl_logA, = ax_logAfit.plot(s_gz, log_angles, 'o')
#    pl_logAfit, = ax_logAfit.plot([], [], 'g', lw=2, label='R')

    text_infos = mpl.figtext(.5, 0.01,'', fontsize=11, ha='center', color='Red')

    def compute_R(model, data):
        sstot = ma.sum( (data - data.mean())**2 )
        ssres = ma.sum( (model - data)**2 )
        R = 1 - ssres/sstot
        return R

    def Curve_zone(ds_start, Arad, cur_tps):
        s = integrate_diff_arc_length(sdx, sdy, cur_tps, scale=scale_cmpix)
        s_gz = s[ds_start:] - s[ds_start]  # s from the beginning of the growth zone
        AA0 = Arad[cur_tps, ds_start:] / Arad[cur_tps, ds_start]
        #Prise en compte des données non masquées
        s_gz = Sc[~AA0.mask]
        signal = ma.log(AA0[:len(s_gz)])
        pl_A.set_data(s_gz, AA0[:len(Sc)])
        pl_logA.set_data(s_gz, signal)

        try:
            ax_Afit.set_xlim(0, s_gz.max())
            ax_Afit.set_ylim(AA0.min(), AA0.max())
            ax_logAfit.set_xlim(0, s_gz.max())
            ax_logAfit.set_ylim(signal.min(), signal.max())
        except:
            pass

        return s_gz, AA0, signal

    def fit_angles(s_gz, log_angles):
        min_func = lambda p, x, y: sum( sqrt( (x*p[0] - y)**2 ) )
        min_func_exp = lambda p, x, y: sum( sqrt( (ma.exp(-x/p[0]) - y)**2 ) )
        #p0 = signal.std()/s_gz[~signal.mask].std()
        #print(p0)

#         opt_params = fmin(min_func, [1.0],
#                           args=(s_gz[~log_angles.mask], log_angles[~log_angles.mask]))

        opt_params_exp = fmin(
                min_func_exp, [1.0],
                args=(s_gz[~log_angles.mask], ma.exp(log_angles[~log_angles.mask])))

        #fitA = poly1d( ma.polyfit(s_gz, log(AA0), 1) )
        #print(opt_params)
        #print(opt_params_exp)
        #print(fitA)
#         Lc = -1/opt_params[0]
        #Lgz = s_gz.max()
        #Si = sinit-sinit[0]
        #Lo = Si.max()
        #GoodL = min((Lgz,Lo))
        #B = GoodL/Lc
#         B = Lgz/Lc
        Lc_exp = opt_params_exp[0]
        #Bexp = GoodL/Lcexp
        Bexp = Lgz/Lc_exp
#         if cur_tps == -1:
#             print_tps = sdx.shape[0]
#         else:
#             print_tps = cur_tps

#         text_infos.set_text("Img: %i, unit: %s, Lgz=%0.2f || fit (A/A0): Lc=%0.2f, B=%0.2f || fit log(A/A0): Lc=%0.2f, B=%0.2f"%(
#             ss_image, length_unit, Lgz, Lc_exp, Bexp, Lc, B))
        text_infos.set_text(
                strings["image"] + " %i, Lgz = %0.2f%s, "%(ss_image+1, Lgz, length_unit)
                + "Lc = %0.2f%s, B=%0.2f"%(Lc_exp, length_unit, Bexp))

        xtmp = linspace(0, max(mpl.gca(). get_xlim()))
        #pl_Afit.set_data(xtmp, exp(-xtmp/Lc))
        pl_logAfit_exp.set_data(xtmp, exp(-xtmp/Lc_exp))

        #Rlogexp = compute_R(ma.exp(-s_gz[~log_angles.mask]/Lc),
        #                    ma.exp(log_angles[~log_angles.mask]))
        Rexp = compute_R(ma.exp(-s_gz[~log_angles.mask]/Lc_exp),
                         ma.exp(log_angles[~log_angles.mask]))
        #pl_Afit.set_label(r'$R^2 = %0.3f$'%Rlogexp)
        pl_logAfit_exp.set_label(r'$R^2 = %0.3f$'%Rexp)
        ax_Afit.legend(loc=0,prop={'size':10})

        #xtmp = linspace(0, max(mpl.gca().get_xlim()))
        #pl_logAfit.set_data(xtmp, -xtmp/Lc)

        #Rlog = compute_R(-s_gz[~log_angles.mask]/Lc, log_angles[~log_angles.mask])
        #pl_logAfit.set_label(r'$R^2 = %0.3f$'%Rlog)
        #ax_logAfit.legend(loc=0, prop={'size':10})

        B_data = {'steady_state_image': ss_image, 'Lc': Lc_exp, 'unit': length_unit,
                  'B': Bexp}

        # Sauvegarde dans le fichier h5
        h5store.save_tige_dict_to_hdf(hdf5file, hdf_tige_id, {'B_data': B_data})

    fit_angles(s_gz, log_angles)

#     mpl.tight_layout()
    grid.tight_layout(fig_B, rect=[0, 0.03, 1, 1])

    # Show the figure
    fig_B.show()


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
        if interekt.get_photo_datetime.get() and dtphoto != []:
            tps = dtphoto
        else:
            tps = arange(len(pictures_names))

        # Boucle sur les tiges
        data_tmp = [None] * len(base_tiges)
        for i in range(len(base_tiges)):

            hdfid = hdf_tiges_id[i]
            tige_x, tige_y, tige_s, tige_taille, tige_angle, tige_tip_angle, tige_zone,\
                    _ = load_postprocessed_data(hdf5file, hdfid)
            tige_R = tiges_data.diam[i]/2.0
            tige_name = tiges_names[i]

            data_tmp[i] = pd.DataFrame({'tige': tige_name, 'angle': tige_tip_angle,
                                        'temps': tps, 'taille': tige_taille,
                                        'rayon': tige_R.mean(1),
                                        'angle_0_360': convert_angle(tige_tip_angle),
                                        'sequence': 0})

        data_out = pd.concat(data_tmp, ignore_index=True)


        # La figure
        legendtige=["tige %s"%str(i) for i in tiges_names]
        fig = mpl.figure(u'Série temporelles avec moyenne',figsize=(10,5))
        grid = mpl.GridSpec(2,4)
        ax1 = mpl.subplot(grid[0,:3])
        if interekt.angle_0_360.get():
            dataa = "angle_0_360"
        else:
            dataa = "angle"


        plot_sequence(data_out, tige_color=tiges_colors, show_lims=False,
                      ydata=dataa, tige_alpha=1.0)
        if not interekt.get_photo_datetime.get():
            mpl.xlabel("Number of images")
        mpl.grid()

        mpl.subplot(grid[1,:3], sharex=ax1)
        yplot = 'taille'
        if scale_cmpix is not None:
            data_out['taille(cm)'] = data_out['taille'] * scale_cmpix
            yplot = 'taille(cm)'

        plot_sequence(data_out, tige_color=tiges_colors, show_lims=False,
                      ydata=yplot, tige_alpha=1)
        ylims = mpl.ylim()
        mpl.ylim(ylims[0]-20, ylims[1]+20)
        if not interekt.get_photo_datetime.get():
            mpl.xlabel("Number of images")
        mpl.grid()

        ax1.legend(legendtige+["Moyenne"], bbox_to_anchor=(1.02, 1), loc=2)

        fig.tight_layout()
        fig.show()

    else:
        print("Pas de serie temporelle, une seule photo traitée")


def save_tige_idmapper():
    """
    Fonction pour sauvegarder les données de post-processing sur les
    estimation des perceptions et autres

    Cette fonction n'est plus appelée. Elle est laissée ici pour documenter
    l'utilisation des anciens fichiers pkl.

    TODO: A supprimer quand le passage a hdf5 est fini
    """
    print('Save data...')

    #Save the tige_id_mapper
    with open(base_dir_path+'tige_id_map.pkl','wb') as f:
        pkl.dump({'tige_id_mapper':tige_id_mapper,
                  'scale_cmpix':scale_cmpix,
                  'growth_data': growth_data,
                  'B_data': B_data,
                  'beta_data': beta_data,
                  'gamma_data': gamma_data,
                  'Tiges_seuil_offset': Tiges_seuil_offset,
                  'Tiges_percent_diam': Tiges_percent_diam}, f)


def measure_pixels():
    global add_dist_draw, dist_measure_pts, plt_measure
    #Function to measure the pixel by drawing two points
    if not add_dist_draw:
        add_dist_draw = True
        dist_measure_pts = [] #empty points list to store them
        try:
            plt_measure.set_data([],[])
            interekt.canvas.draw()
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
            interekt.canvas.draw()
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
    topcal.title(_("Scale"))

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

    Label(calframe, text='pixel:').pack()
    pixel_distance.pack()
    Label(calframe, text='cm:').pack()
    cm_distance.pack()
    calframe.pack()

    calbutton_calibration = Button(master=topcal, text=_("Measure distance"),
                                   command=measure_pixels)
    calbutton_calibration.pack(fill=Tk.X)

    calbutton_updatecalibration = Button(master=topcal, text=_("Update scale"),
                                         command=update_scale_pxcm)
    calbutton_updatecalibration.pack(fill=Tk.X)


def detection_step_setting():
    global tk_detection_step

    # Ajout d'une boite tk pour l'export
    top_step = Tk.Toplevel(master=root)
    top_step.title(_("Skeleton detection step"))

    step_frame = Frame(top_step)

    w1 = Label(step_frame, text=_("Size of the detection step (in pixels):"))
    w1.pack(fill='x', expand=True)

    tk_detection_step = Tk.DoubleVar()

    # Retrieve the detection step from the h5 file
    detection_step = h5store.get_detection_step(hdf5file)

    if detection_step is not None:
        tk_detection_step.set(detection_step)
    else:
        tk_detection_step.set(DETECTION_STEP_DEFAULT)

    w2 = Tk.Scale(step_frame, from_=0.1, to=5, resolution=0.01, variable=tk_detection_step,
                  orient=Tk.HORIZONTAL)
    w2.pack(fill='x', expand=True)

    step_frame.pack()


def select_steady_state_image():
    """Top window for selecting the steady-state image.

    The steady-state image is selected from the image liste.

    In addition, there is a checkbox for removing this image from times
    series, which can be useful when the steady-state image has been taken
    a long time after other images.
    """
    # Create Tkinter top window
    top_steadystate = Tk.Toplevel(master=root)
    top_steadystate.title(_("Steady-state image"))

    list_frame = Frame(top_steadystate)
    topsbar = Tk.Scrollbar(list_frame, orient=Tk.VERTICAL)
    listb = Tk.Listbox(master=list_frame, yscrollcommand=topsbar.set)
    topsbar.config(command=listb.yview)
    topsbar.pack(side=Tk.RIGHT, fill=Tk.BOTH)

    images_names = h5store.get_images_names(hdf5file)
    for image_name in images_names:
        listb.insert(Tk.END, image_name)

    # If a steady-state image has already been selected
    if interekt.steady_state_image is not None:
        listb.selection_set(interekt.steady_state_image)
        listb.see(interekt.steady_state_image)

    # Function to call when an image has been selected
    listb.bind("<<ListboxSelect>>", on_select_steady_state_image)

    listb.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
    list_frame.pack(fill=Tk.BOTH, expand=1)

    # Choose whether the steady-state image should be excluded from time series.
    # This can be useful when the steady-state image has been taken a long time after
    # other images.
    check_frame = Frame(top_steadystate)
    rm_from_time_series_checkbox = Checkbutton(
            check_frame, width=30,
            text=_("Exclude image from time series"),
            variable=interekt.exclude_steady_state_from_time_series,
            onvalue=True, offvalue=False)
    rm_from_time_series_checkbox.pack()
    check_frame.pack(fill=Tk.X, side=Tk.BOTTOM)

    Tk.Button(check_frame, text=_("Save"),
              command=save_steady_state).pack(fill=Tk.X, side=Tk.BOTTOM, expand=True)


def on_select_steady_state_image(event):
    """Called when an image has been selected from the list as the
    steady-state image."""
    sender = event.widget
    try:
        idx = sender.curselection()[0]
    except:
        idx = sender.curselection()
    if idx == interekt.steady_state_image:
        sender.selection_clear(idx)
        interekt.steady_state_image = None
    else:
        interekt.steady_state_image = idx
        try:
            interekt.plot_image(interekt.steady_state_image, keep_zoom=True)
        except Exception as e:
            print("Image loading error!!!")
            print(e)


def save_steady_state():
    """Save the selected steady-state image in the h5 file.

    Save also whether the steady-state image has to be excluded from
    time series.
    """
    print("Save ss image ", interekt.steady_state_image, \
            interekt.exclude_steady_state_from_time_series)
    h5store.save_steady_state(
            hdf5file,
            interekt.steady_state_image,
            interekt.exclude_steady_state_from_time_series)


def find_dict_subkeys(input_dict):
    """"Find keys of a dictionary nested in another dictionary.

    Example
    -------
    data = {"0": {"a":0, "b":1},
            "1": {"a": 10, "b": 12}}

    find_dict_subkeys(data)

    Returns: ["a", "b"]
    """
    dict_keys = []
    for key in list(input_dict):
        if input_dict[key] is not None:
            dict_keys = input_dict[key].keys()
            break

    return dict_keys


def _export_tige_id_mapper_to_csv(summary=True):
    """Export phenotyping data into CSV file."""

    proposed_filename = "Phenotype_gravi_proprio.csv"
    outfileName = filedialog.asksaveasfilename(
            parent=root,
            filetypes=[("Comma Separated Value, csv","*.csv")],
            title=_("Export phenotyping data"),
            initialfile=proposed_filename,
            initialdir=base_dir_path)

    if len(outfileName) > 0:

        # On récupère les données dans le fichier hdf5
        growth_data = h5store.get_postprocessing(hdf5file, 'growth_data')
        beta_data = h5store.get_postprocessing(hdf5file, 'beta_data')
        gamma_data = h5store.get_postprocessing(hdf5file, 'gamma_data')
        B_data = h5store.get_postprocessing(hdf5file, 'B_data')
        tiges_names = h5store.get_tiges_names(hdf5file)
        hdf_tiges_id = h5store.get_tiges_indices(hdf5file)

        # Il faut trouver les clés des données standards pour chaque dico
        growth_keys = find_dict_subkeys(growth_data)
        beta_keys = find_dict_subkeys(beta_data)
        gamma_keys = find_dict_subkeys(gamma_data)
        B_keys = find_dict_subkeys(B_data)

        scale_cmpix = h5store.get_pixelscale(hdf5file)

        ntiges = len(base_tiges)
        output = pd.DataFrame()
        column_names = []  # to keep column name order
        if summary:
            for i in range(ntiges):
                output_i = OrderedDict()
                output_i['tige_name'] = tiges_names[i]

                hdf_id = hdf_tiges_id[i]

                growth_dict = growth_data.get(hdf_id)
                beta_dict = beta_data.get(hdf_id)
                gamma_dict = gamma_data.get(hdf_id)
                B_dict = B_data.get(hdf_id)

                if beta_dict is not None and "beta_tilde" in beta_dict:
                    output_i["beta_tilde"] = beta_dict["beta_tilde"]
                    output_i["beta_fit_R2"] = beta_dict["R2"]

                if gamma_dict is not None and "gamma" in gamma_dict:
                    if python2:
                        gamma_unit = str(gamma_dict["gamma_unit"])
                    else:
                        gamma_unit = str(gamma_dict["gamma_unit"], "utf8")
                    output_i["gamma_(%s)"%gamma_unit] = gamma_dict["gamma"]
                    output_i["gamma_fit_R2"] = gamma_dict["R2"]
                    gamma_tilde = gamma_dict.get("gamma_tilde")
                    if gamma_tilde is not None:
                        output_i["gamma_tilde"] = gamma_tilde

                if B_dict is not None and 'B' in B_dict:
                    B = B_dict["B"]
                    output_i["B"] = B
                    if gamma_dict is None or "gamma_tilde" not in gamma_dict:
                        # Try computing gamma_tilde from B and beta_tilde
                        if beta_dict is not None and "beta_tilde" in beta_dict:
                            beta_tilde = beta_dict["beta_tilde"]
                            radius = beta_dict["radius"]
                            Lgz = growth_dict["Lgz"]
                            output_i["gamma_tilde"] = beta_tilde * Lgz / (B * radius)
                    output_i["Lc"] = B_dict["Lc"]

                if growth_dict is not None and "dLdt" in growth_dict:
                    if python2:
                        dLdt_unit = str(growth_dict["dLdt_unit"])
                    else:
                        dLdt_unit = str(growth_dict["dLdt_unit"], "utf8")
                    output_i["dLdt_(%s)"%dLdt_unit] = growth_dict["dLdt"]
                    if "Lgz" in growth_dict:
                        if python2:
                            length_unit = str(growth_dict["length_unit"])
                            RER_unit = str(growth_dict["RER_unit"])
                        else:
                            length_unit = str(growth_dict["length_unit"], "utf8")
                            RER_unit = str(growth_dict["RER_unit"], "utf8")
                        output_i["Lgz_(%s)"%length_unit] = growth_dict["Lgz"]
                        output_i["RER_(%s)"%RER_unit] = growth_dict["RER"]

                if beta_dict is not None and "beta_tilde" in beta_dict:
                    output_i["radius_(%s)"%length_unit] = beta_dict["radius"]
                    output_i["Ainit_(rad)"] = beta_dict["Ainit"]
                    if python2:
                        dAdt_unit = str(beta_dict["dAdt_unit"])
                    else:
                        dAdt_unit = str(beta_dict["dAdt_unit"], "utf8")
                    output_i["dAdt_(%s)"%dAdt_unit] = beta_dict["dAdt"]

                output_i['scale (cm/pix)'] = scale_cmpix

                new_column_names = list(output_i.keys())
                if len(new_column_names) > len(column_names):
                    column_names = new_column_names

                output = output.append(output_i, ignore_index=True)
        else:
            try:
                for i in range(ntiges):
                    output_i = OrderedDict()
                    output_i['scale (cm/pix)'] = scale_cmpix
                    output_i['id_tige'] = hdf_tiges_id[i]
                    output_i['tige_name'] = tiges_names[i]

                    hdf_id = hdf_tiges_id[i]

                    if hdf_id in growth_data:
                        for key in growth_keys:
                            if growth_data[hdf_id] is None:
                                output_i["growth_%s" % key] = None
                            else:
                                output_i["growth_%s" % key] = growth_data[hdf_id][key]

                    if hdf_id in beta_data:
                        for key in beta_keys:
                            if beta_data[hdf_id] is None:
                                output_i["beta_%s" % key] = None
                            else:
                                output_i["beta_%s" % key] = beta_data[hdf_id][key]

                    if hdf_id in gamma_data:
                        for key in gamma_keys:
                            if gamma_data[hdf_id] is None:
                                output_i["gamma_%s" % key] = None
                            else:
                                output_i["gamma_%s" % key] = gamma_data[hdf_id][key]

                    if hdf_id in B_data:
                        for key in B_keys:
                            if B_data[hdf_id] is None:
                                output_i["B_%s" % key] = None
                            else:
                                output_i["B_%s" % key] = B_data[hdf_id][key]

                    new_column_names = list(output_i.keys())
                    if len(new_column_names) > len(column_names):
                        column_names = new_column_names

                    output = output.append(output_i, ignore_index=True)
            except Exception as e:
                    print("Error during data export.\nError (tige %i): %s"%(i, e))

        output = output.reindex(columns=column_names)
        output.to_csv(outfileName, index=False)
        print("Saved to %s"%outfileName)


###############################################################################

############################# Interekt class ##################################

class Interekt:

    def __init__(self, master):
        self.master = master
        self.master.style = Style()
        self.master.style.theme_use("clam")
        title = "Interekt -- " + _("version") + ": %s"%(__version__)
        self.master.wm_title(title)
        print(title)

        #TOP MENU BAR
        menubar = Tk.Menu(self.master)

        # Export menu
        exportmenu = Tk.Menu(menubar, tearoff=0)

        exportmenu.add_command(
                label=_("Average time series"),
                #label=_("Série temporelle moyenne"),
                command=_export_mean_to_csv)

        exportmenu.add_command(
                label=_("Time series for each organ"),
                #label=_("Séries temporelles par organe"),
                command=_export_meandata_for_each_tiges)

        exportmenu.add_command(
                label=_("Global time series (tip angle and length)"),
                #label=_("Séries temporelles globales (angle au bout et longueur)"),
                command=_export_to_csv)

        exportmenu.add_command(
                label=_("Global time series + skeleton"),
                #label=_("Séries temporelles globales + squelette"),
                command=_export_xytemps_to_csv)

        exportmenu.add_command(
                label=_("Phenotype (graviception, proprioception)"),
                #label=_("Phénotype (graviception, proprioception)"),
                command=_export_tige_id_mapper_to_csv)

        menubar.add_cascade(label=_("Export"), menu=exportmenu)

        #Plot menu
        plotmenu = Tk.Menu(menubar, tearoff=0)
        plotmenu.add_command(label=_("Time series"), command=plot_moyenne)
        menubar.add_cascade(label=_("Plots"), menu=plotmenu)

        #options menu
        options_menu = Tk.Menu(menubar)

        #Pour chercher le temps dans les données EXIFS des images
        self.get_photo_datetime = Tk.BooleanVar()
        self.get_photo_datetime.set(True)
        options_menu.add_checkbutton(label=_("Extract photo time"),
                onvalue=True, offvalue=False, variable=self.get_photo_datetime)

        # If we want to keep angles between 0 and 360 degrees
        self.angle_0_360 = Tk.BooleanVar()
        self.angle_0_360.set(False)
        options_menu.add_checkbutton(label=_("Angle modulo 360 (0->360)"),
                onvalue=True, offvalue=False, variable=self.angle_0_360)

        #check_black = Tk.BooleanVar()
        #check_black.set(False)
        #options_menu.add_checkbutton(label="Photos noires (<5Mo)", onvalue=True, offvalue=False, variable=check_black)
        # options_menu.add_command(label="Test detection", command=test_detection) # TODO: BUG WITH THREAD

        # If we want to select an image as steady state
        self.steady_state_image = None  # number of the image at steady state
        self.exclude_steady_state_from_time_series = Tk.BooleanVar()
        self.exclude_steady_state_from_time_series.set(False)
        options_menu.add_command(
                label=_("Select steady-state image"),
                command=select_steady_state_image)

        # Set the scale
        options_menu.add_command(label=_("Scale"), command=pixel_calibration)

        # Set the space step for skeleton detection process
        options_menu.add_command(label=_("Step for skeleton detection"), command=detection_step_setting)

        #TODO: Pour trier ou non les photos
        #sort_photo_num = Tk.BooleanVar()
        #sort_photo_num.set(True)
        #options_menu.add_checkbutton(label="Sort pictures", onvalue=True, offvalue=False, variable=sort_photo_num)

        menubar.add_cascade(label=_('Options'), menu=options_menu)

        # Menu for choosing the model to use
        model_menu = Tk.Menu(menubar)
        self.model = Tk.StringVar()
        self.model.set("No model")

        model_menu.add_radiobutton(label=_('No model'), variable=self.model,
                                   value="No model")

        model_menu.add_radiobutton(label=_('Linear growth'), variable=self.model,
                                   value="Linear growth")

#         model_menu.add_radiobutton(label='AC', variable=self.model,
#                                    value="AC")

        model_menu.add_radiobutton(label='ACĖ', variable=self.model,
                                   value="ACE")

#        model_menu.add_radiobutton(label='ACα', variable=self.model,
#                                   value="ACalpha")

        menubar.add_cascade(label=_('Model'), menu=model_menu)

        #Display the menu
        self.master.config(menu=menubar)
        self.master.columnconfigure(1, weight=1, minsize=600)
        self.master.rowconfigure(1, weight=1)

        #BOTTOM MENU BAR
        buttonFrame = Frame(master=self.master)
        #buttonFrame.pack(side=Tk.BOTTOM)
        buttonFrame.grid(row=0, column=0, sticky=Tk.W)
        self.button_ouvrir = Button(
                master=buttonFrame, text=_("Open"), command=self._open_files)
        self.button_listimages = Button(
                master=buttonFrame, text=_("Image list"),  # "Liste d'images"
                command=show_image_list, state=Tk.DISABLED)
        self.button_addtige = Button(
                master=buttonFrame, text=_("Add a base"),  # 'Ajouter une base'
                command=self._addtige, state=Tk.DISABLED)
        self.button_suppr_all_tiges = Button(
                master=buttonFrame, text=_("Suppress all bases"),  # 'Supprimer les bases'
                command=self._suppr_all_tiges, state=Tk.DISABLED)
        self.button_traiter = Button(
                master=buttonFrame, text=_('Process'),  # 'Traiter'
                command=launch_process, state=Tk.DISABLED)
        self.button_addmark = Button(
                master=buttonFrame, text=_("Add a mark"),  # 'Ajouter une marque'
                command=self._addmark, state=Tk.DISABLED)
        self.button_suppr_all_marks = Button(
                master=buttonFrame, text=_("Suppress all marks"),  # 'Supprimer les marques'
                command=self._suppr_all_marks, state=Tk.DISABLED)

        # Progress bar
        self.prog_bar = Progressbar(master=self.master, mode='determinate')
        #Ajout d'un bouton export to csv
        #button_export = Tk.Button(master=buttonFrame, text=u'Exporter série vers (csv)', command=_export_to_csv, state=Tk.DISABLED)

        self.button_ouvrir.grid(row=0, column=0)
        self.button_listimages.grid(row=0, column=1)
        self.button_addtige.grid(row=0, column=2)
        self.button_suppr_all_tiges.grid(row=0, column=3)
        self.button_traiter.grid(row=0, column=4)
        self.button_addmark.grid(row=0, column=5)
        self.button_suppr_all_marks.grid(row=0, column=6)

        self.prog_bar.grid(row=2, columnspan=2, sticky=Tk.E+Tk.W)

        #Floating menu (pour afficher des infos pour une tige)
        self.floatmenu = Tk.Menu(self.master, tearoff=0)

        #floatmenu.add_command(label="Inverser la base", command=_reverse_tige)

        self.floatmenu.add_command(label=_("Time series"),
                                   command=show_time_series)

        self.floatmenu.add_command(label=strings["Spatiotemporal heatmaps"],
                                   command=show_heatmaps)

        self.floatmenu.add_command(label=_("Angles and curvatures"),
                                   command=show_angle_and_curvature)

        self.floatmenu.add_separator()

        self.floatmenu.add_command(label=_("Settings"),
                                   command=show_tige_options)

        self.floatmenu.add_command(label=_("Suppress the base"),
                                   command=remove_tige)

        self.floatmenuisopen = False

        #button_export.pack(side=Tk.LEFT)
        #figsize=(10,8)
        self.fig = mpl.figure()
        self.ax = self.fig.add_subplot(111)

        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        try:
            self.canvas.draw()  # Python 3
        except:
            self.canvas.show()  # Python 2
        #canvas.get_tk_widget().pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)
        self.canvas.get_tk_widget().grid(row=1, columnspan=2, sticky=Tk.W+Tk.E+Tk.N+Tk.S)
        tbar_frame = Tk.Frame(self.master)
        tbar_frame.grid(row=0, column=1, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tbar_frame)
        self.toolbar.update()
        #canvas._tkcanvas.pack(side=Tk.BOTTOM, fill=Tk.BOTH, expand=1)

        # La variable globale update_imgid permet de stocker le after de
        # tkinter qui met a jour la figure en fullresolution et de pouvoir
        # l'annuler si on change l'image trop vite.
        self.update_imgid = None

        #On affiche la première image
        self.plot_image(cur_image)
        self.canvas._tkcanvas.config(cursor='cross')

        # Connxion de fonctions retours au canvas matplotlib
        cidkey = self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.canvas.mpl_connect('button_press_event', self.onClick)
        self.canvas.mpl_connect('pick_event', self.onPick)
        self.canvas.mpl_connect("motion_notify_event", self.onmousemove)

    def update_floatmenu(self, tige_hdf_id, line_type):
        """Set the available items in floatmenu depending on the context.

        The method is called when a tige or a base has been picked, just before the
        floating menu pops up. It populates the floating menu associated with a tige with
        the actions which are possible given the model chosen and the values already
        estimated.

        Parameters:
        - tige_id_hdf: tige identifier in the hdf file
        - line_type: type of the clicked line ('tige' or 'base')
        """
        self.floatmenu.delete(0, "end")  # reset the floating menu
        if line_type == "tige":
            self.floatmenu.add_command(label=_("Time series"),
                                       command=show_time_series)
            self.floatmenu.add_command(label=strings["Spatiotemporal heatmaps"],
                                       command=show_heatmaps)
            self.floatmenu.add_command(label=_("Angles and curvatures"),
                                       command=show_angle_and_curvature)
            self.floatmenu.add_separator()

            growth_data = h5store.get_postprocessing(hdf5file, 'growth_data', tige_hdf_id)

            model = self.model.get()

            if model == "Linear growth" or model == "ACE":
                # Estimation of the growth rate and the growth length
                self.floatmenu.add_command(label=_("Estimate growth rate"),
                                           command=show_growth_rate)
                if growth_data is not None and growth_data.get("dLdt") is not None:
                    # The growth rate has to be estimated before the growth length
                    self.floatmenu.add_command(label=_("Estimate growth length"),
                                               command=show_growth_length)
                self.floatmenu.add_separator()

            if (model == "ACE" and growth_data is not None
                    and growth_data.get("dLdt") is not None):
                # Estimation of beta tilde and gamma tilde
                # The growth rate has to be estimated before beta tilde
                self.floatmenu.add_command(label=_("Estimate β̃ (beta tilde)"),
                                           command=show_beta_tilde)
                if growth_data.get("Lgz") is not None:
                    # The growth length has to be estimated before gammatilde
                    self.floatmenu.add_command(label=_("Estimate γ̃ (gamma tilde)"),
                                               command=show_gamma_tilde)
                    self.floatmenu.add_command(label=_("Estimate B"),
                                               command=show_B)
                self.floatmenu.add_separator()

        # In all cases, add the base commands
        self.floatmenu.add_command(label=_("Settings"),
                                   command=show_tige_options)
        self.floatmenu.add_command(label=_("Suppress the base"),
                                   command=remove_tige)

    def popup(self, tige_id):
        global cur_tige
         # display the popup menu
        cur_tige = tige_id

        try:
            self.floatmenu.tk_popup(int(self.master.winfo_pointerx()),
                               int(self.master.winfo_pointery()))
            self.floatmenuisopen = True
        finally:
            # make sure to release the grab (Tk 8.0a1 only)
            self.floatmenu.grab_release()
            #pass

    def on_key_event(self, event):
        global nbclick, add_tige, add_mark
        print('you pressed %s'%event.key)
        key_press_handler(event, self.canvas, self.toolbar)

        if event.key == '+':
            add_tige = True
            nbclick = 0

        if event.key == 'right':
            if cur_image + 1 < len(h5store.get_images_names(hdf5file)):
                self.plot_image(cur_image + 1, keep_zoom=True)

        if event.key == 'left':
            if cur_image > 1:
                self.plot_image(cur_image - 1, keep_zoom=True)

        if event.key == 'escape':
            # Cancel add_tige
            if add_tige:
                add_tige = False
                if nbclick == 1:
                    nbclick = 0
                    base_tiges.pop(-1)
                self.plot_image(cur_image, keep_zoom=True)
                self.change_button_state()
            # Cancel add_mark
            elif add_mark:
                add_mark = False
                if nbclick == 1:
                    nbclick = 0
                    marks.pop(-1)
                self.plot_image(cur_image, keep_zoom=True)
                self.change_button_state()

    def onClick(self, event):
        global base_tiges, marks, nbclick, add_tige, add_mark
        global plt_measure, add_dist_draw, dist_measure_pts, pixel_distance, used_scale

        # Restore focus on the current canvas
        self.canvas.get_tk_widget().focus_force()

        if event.button == 1:
            # Manage how to add a tige
            if add_tige:
                x, y = event.xdata, event.ydata
                if x != None and y != None:
                    # On récupère la résolution de l'image affiché
                    scale = used_scale
                    xyscale = [x*scale, y*scale]
                    if nbclick == 0:
                        base_tiges.append([xyscale])
                        nbclick += 1
                        self.ax.plot(x, y, 'r+')
                        self.canvas.draw()
                    else:
                        base_tiges[-1].append(xyscale)
                        nbclick = 0
                        add_tige = False

                        # Création de la tige dans le fichier hdf5.
                        # On crée un id qui est plus grand que l'id max
                        # qui se trouve déjà dans le fichier hdf5.
                        tiges_ids = h5store.get_tiges_indices(hdf5file)
                        if tiges_ids == []:
                            # c'est la première base
                            new_tige_id = 0
                        else:
                            # On ajoute 1 au max
                            new_tige_id = max(tiges_ids) + 1

                        h5store.create_tige_in_hdf(hdf5file, new_tige_id)

                        # Besoin de sauvegarder les points de base de la tige
                        h5store.save_base_tige(hdf5file, new_tige_id, base_tiges[-1])

                        # Besoin de crée un nom pour la tige (ancien
                        # role du tige_id_mapper). On lui donne le nom
                        # qui correspond a son tige_id
                        h5store.save_tige_name(hdf5file, new_tige_id, str(new_tige_id))

                        # On recharge les données des tiges dans le GUI
                        load_hdf5_tiges(hdf5file, display_message=False)

                        # Tracé des bases de tiges
                        self.plot_basetiges(force_redraw=True)

                    self.change_button_state()

            elif add_mark:
                x, y = event.xdata, event.ydata
                if x != None and y != None:
                    # On récupère la résolution de l'image affiché
                    scale = used_scale
                    xy_scaled = scale * array([x, y])
                    if nbclick == 0:
                        mark = {"name": len(marks) + 1,
                                "image": cur_image,
                                "start": xy_scaled}
                        if self.get_photo_datetime.get() and dtphoto != []:
                            # Time in minutes
                            t = int((dtphoto[cur_image] - dtphoto[0]).total_seconds()
                                    / 60.)
                            mark["time"] = t
                        marks.append(mark)
                        nbclick += 1
                        self.ax.plot(x, y, 'b+')
                        self.canvas.draw()
                    else:
                        mark = marks[-1]
                        mark["end"] = xy_scaled
                        nbclick = 0
                        add_mark = False
                        if not self.find_mark_tige_intersection(mark):
                            marks.pop()
                        else:
                            # TODO: Création de la marque dans le fichier hdf5.
                            # Tracé des marques
                            self.plot_marks(force_redraw=True)

                    self.change_button_state()

            if add_dist_draw:
                if dist_measure_pts == []:
                    plt_measure, = self.ax.plot(event.xdata,event.ydata, 'yo-',
                                                label='measure', zorder=10)
                    dist_measure_pts += [event.xdata, event.ydata]
                    self.canvas.draw_idle()
                else:
                    plt_measure.set_data((dist_measure_pts[0], event.xdata),
                                         (dist_measure_pts[1], event.ydata))
                    self.canvas.draw_idle()
                    if pixel_distance != None:
                        tmpd = sqrt((dist_measure_pts[0] - event.xdata)**2
                                    + (dist_measure_pts[1] - event.ydata)**2)
                        pixel_distance.delete(0, Tk.END)
                        pixel_distance.insert(0, str('%0.2f'%tmpd))

                    dist_measure_pts = []
                    add_dist_draw = False

            #Close floatmenu
            if self.floatmenuisopen:
                self.floatmenu.unpost()
                self.floatmenuisopen = False

        if event.button == 3:

            # Cancel add_tige
            if add_tige:
                add_tige = False
                if nbclick == 1:
                    nbclick = 0
                    base_tiges.pop(-1)
                self.plot_image(cur_image, keep_zoom=True, force_clear=False)
                self.change_button_state()

            # Cancel add_mark
            elif add_mark:
                add_mark = False
                if nbclick == 1:
                    nbclick = 0
                    marks.pop(-1)
                self.plot_image(cur_image, keep_zoom=True, force_clear=False)
                self.change_button_state()


    def onPick(self, event):

        if isinstance(event.artist, mpl.Line2D):
            thisline = event.artist
            label = thisline.get_label()

            # Besoin de savoir si c'est la base ou la tige qui a été pickée
            # pour éviter les ouvertures à répétition de la même fenêtre.
            # Si c'est une marque, on arrête tout.

            if 'mark_' in label:
                line_type = "mark"
            elif 'base_' in label:
                line_type = "base"
                label = label.replace('base_','')
            else:
                line_type = "tige"

            if line_type == "base" or line_type == "tige":
                tige_id = int(label)
                tige_hdf_id = get_hdf5_tigeid(hdf5file, tige_id)
                try:
                    print(u'Selection de la tige %s' % tiges_names[tige_id])
                    print(u'Enregistrée dans le fichier hdf sous tige%i ' % tige_hdf_id)
                    print(u'Position dans la liste du gui %i' % tige_id)
                except:
                    print(u'Selection de la tige %i du gui'%(tige_id))
                    print(u'Enregistrée sous le nom tige%i dans le fichier hdf' % tige_id_hdf)

            if event.mouseevent.button == 1:  # left click
                if line_type == "tige":
                    show_time_series(tige_id=tige_id)
                elif line_type == "mark":
                    mark_number = int(label.split("_")[-1])
                    # no action for now

            if event.mouseevent.button == 3:  # right click
                if not self.floatmenuisopen and (line_type == "tige"
                                                 or line_type == "base"):
                    self.update_floatmenu(tige_hdf_id, line_type)
                    self.popup(tige_id)


    def onmousemove(self, event):
        """
        Method called each time the mouse moves within the Matplotlib canvas.
        It changes the mouse pointer icon depending on the current mode.
        """
        # for Interekt-specific actions
        if not self.toolbar.mode:  # no zoom nor span
            if add_tige:
                self.canvas._tkcanvas.config(cursor='crosshair')
            else:
                self.canvas._tkcanvas.config(cursor='arrow')

        # if zoom button pressed, set a 'resize' icon
        elif self.toolbar.mode == 'zoom rect':
            self.canvas._tkcanvas.config(cursor='sizing')

        # if pan button pressed, set a hand icon
        elif self.toolbar.mode == 'pan/zoom':
            self.canvas._tkcanvas.config(cursor='hand2')


    def change_button_state(self):

        if len(marks) > 0:
            self.button_suppr_all_marks.config(state=Tk.NORMAL)
        else:
            self.button_suppr_all_marks.config(state=Tk.DISABLED)

        if len(base_tiges) > 0:
            for bt in [self.button_suppr_all_tiges,
                       self.button_traiter,
                       self.button_addmark]:
                bt.config(state=Tk.NORMAL)
        else:
            for bt in [self.button_suppr_all_tiges,
                       self.button_traiter,
                       self.button_addmark]:
                bt.config(state=Tk.DISABLED)

        if len(h5store.get_images_names(hdf5file)) > 0:
            for bt in [self.button_addtige, self.button_listimages]:
                bt.config(state=Tk.NORMAL)
        else:
            for bt in [self.button_addtige,
                       self.button_suppr_all_tiges,
                       self.button_traiter,
                       self.button_addmark,
                       self.button_listimages]:
                bt.config(state=Tk.DISABLED)


    def display_message(self, message, msg_color='red'):
        """Display a message in the center of the Matplotlib canvas.

        Parameters
        ----------

        message, unicode:
            Message a afficher

        msg_color, string optional:
            Permet de définir la couleur du message affiché
        """
        if python2:
            try:
                message = unicode(message, "utf-8")
            except:
                pass
        self.ax.clear()
        self.ax.text(0.5, 0.5, message,
                    ha='center', va='center', color=msg_color,
                    transform=self.ax.transAxes)
        self.canvas.draw()


    ########################### IMAGE PLOTTING ###############################

    def plot_image_fullres(self):
        """
        Affiche l'image en pleine résolution. Cette fonction est lancée par
        la méthode after de tkinter pour améliorer la résolution de l'image
        quand on reste dessus.
        """
        global img_object

        # On charge l'image en basse résolution
        if cur_image is not None:
            self.plot_image(cur_image, keep_zoom=True, image_resolution=0)

        # Remet l'id pour tk after a None
        self.update_imgid = None

    #global pour sauvegarde la résolution utilisée cela permet de faire
    #suivre les xlims ylims de la figure correctement
    used_scale = 1
    def plot_image(self, img_num, keep_zoom=False, force_clear=False, image_resolution=1):
        global cur_image, btige_plt, btige_text, mark_plt, mark_text, img_object, text_object
        global used_scale

        # Doit on annuler la mise a jour de l'image fullresolution
        if self.update_imgid is not None:
            self.master.after_cancel(self.update_imgid)
            self.update_imgid = None

        # Met a jour la variable globale cur_image
        cur_image = img_num

        # Doit on garder le zoom en mémoire pour le restaurer à la fin
        if keep_zoom:
            oldxlims = self.ax.get_xlim()
            oldylims = self.ax.get_ylim()

        # Doit-on tout nettoyer
        if force_clear:
            self.ax.clear()

            # Reset btige
            btige_plt = [None] * len(base_tiges)
            btige_text = [None] * len(base_tiges)

            # Reset marks
            mark_plt = [None] * len(marks)
            mark_text = [None] * len(marks)

        if img_num is None:
            self.ax.axis('off')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.display_message(_("Load images or result file"))

        else:
            self.ax.axis('on')

            # On charge l'image en petite résolution
            imtmp = h5store.open_hdf5_image(hdf5file, img_num,
                                            resolution=image_resolution)

            scalefactors = h5store.get_image_scale(hdf5file,
                                                   resolution=image_resolution)

            # Changer le titre de la fenetre principal pour afficher la photo
            try:
                self.master.wm_title("Interekt | %s"%(h5store.get_images_names(hdf5file,
                                                                               img_num)))
            except Exception as e:
                print("""Erreur dans le changement de nom de la fenêtre""")
                print(e)

            # Mise a jour de la position sélectionnée dans la liste des
            # images si elle existe (si la fenetre "tk_toplevel_listimages"
            # est ouverte)
            if tk_list_images:
                tk_list_images.selection_clear(0, Tk.END)
                tk_list_images.selection_set(img_num)
                tk_list_images.see(img_num)

            if img_object is None:
                print(imtmp.shape)
                if imtmp.shape[2] < 3:
                    img_object = self.ax.imshow(imtmp, cmap=mpl.cm.gray)
                else:
                    img_object = self.ax.imshow(imtmp)
            else:
                #print shape(imtmp)
                img_object.set_data(imtmp)
                # Mise a jour de l'extent de l'imshow
                img_object.set_extent((0, imtmp.shape[1], imtmp.shape[0], 0))

            # Plot des bases de tige et des marques lagrangiennes
            image_scale = 1/float(scalefactors[0])
            self.plot_basetiges(ratio=image_scale)
            self.plot_marks(ratio=image_scale)

            if image_resolution > 0:
                # On decime le nombre de points sur les tiges pour augmenter la vitesse d'affichage
                tige_plot_decimation = 100
            else:
                # Quand on est en fullresolution on decime moins les tiges
                tige_plot_decimation = 10

            if tiges_data is not None:
                # Plot des tiges traitée
                for ti, tige_xyplot in enumerate(tiges_plot_object):
                    # On teste d'abord si il y a des données
                    tigeh5id = get_hdf5_tigeid(hdf5file, ti)

                    if h5store.is_data(hdf5file, 'xc', tigeh5id):

                        # Creation des graphiques
                        if tige_xyplot is None:
                            # Les tiges sont tracées avec une décimation définie
                            # avec le scalefactor
                            try:
                                tmp, = self.ax.plot(
                                        tiges_data.xc[ti,int(img_num),::tige_plot_decimation].T * image_scale,
                                        tiges_data.yc[ti,int(img_num),::tige_plot_decimation].T * image_scale,
                                        lw=3, picker=5, label="%i"%(ti), color=tiges_colors[ti])
                                tiges_plot_object[ti] = tmp
                            except Exception as e:
                                print(u'Erreur de chargement des données pour la tige %s en position %i du gui'%(tiges_names[ti], ti))
                                print(e)

                        # Mise a jour des graphiques
                        else:
                            try:
                                tiges_plot_object[ti].set_data(
                                        tiges_data.xc[ti,int(img_num),::tige_plot_decimation].T * image_scale,
                                        tiges_data.yc[ti,int(img_num),::tige_plot_decimation].T * image_scale)
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

                self.ax.set_xlim(oldxlims)
                self.ax.set_ylim(oldylims)

        self.canvas.draw()
        print(u'fin de mise à jour de la figure à la résolution %i' % image_resolution)

        # Mise a jour de la memoire de la résolution utilisée
        if img_num is not None:
            used_scale = scalefactors[0]
        # On lance un after si pas de changement d'image et que la
        # résolution est une résolution plus faible que l'image originale
        if img_num is not None and image_resolution > 0:
            self.update_imgid = self.master.after(350, self.plot_image_fullres)


    ####################### GESTION GRAPHIQUE DES TIGES #######################

    def _addtige(self):
        """Activates the addition of a tige with the GUI.

        See below the handling if click events.
        """
        global add_tige, nbclick, btige_plt, btige_text
        add_tige = True
        nbclick = 0
        self.change_button_state()


    def _suppr_all_tiges(self):
        """Suppress all tiges in the hdf5 file."""
        reset_graph_data()

        tiges_ids_hdf = h5store.get_tiges_indices(hdf5file)

        for idt in tiges_ids_hdf:
            h5store.delete_tige(hdf5file, idt)

        # Recharge les données depuis le fichier hdf
        load_hdf5_tiges(hdf5file)
        # Replot the current image
        self.plot_image(cur_image, force_clear=True)
        self.change_button_state()

    def plot_basetiges(self, force_redraw=False, ratio=1):
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

        oldxlims = self.ax.get_xlim()
        oldylims = self.ax.get_ylim()

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
                tige_name = tiges_names[i]
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

                if len(btige_plt) <= i:
                    btige_plt += [None]
                if len(btige_text) <= i:
                    btige_text += [None]

                if btige_plt[i] is None:
                    btige_plt[i], = self.ax.plot(
                            [base[0][0] * ratio, base[1][0] * ratio],
                            [base[0][1] * ratio, base[1][1] * ratio],
                            symb, label='base_%i'%i, lw=1.1, picker=5)
                else:
                    btige_plt[i].set_data([base[0][0] * ratio, base[1][0] * ratio],
                                          [base[0][1] * ratio, base[1][1] * ratio])

                # Dessin de la normale
                theta = arctan2(base[1][1] - base[0][1], base[1][0] - base[0][0])
                L = 0.25 * sqrt((base[1][1] - base[0][1])**2
                                + (base[1][0]-base[0][0])**2
                               ) * ratio     # Taille en pixel de la normale
                xc = 0.5 * (base[1][0] + base[0][0]) * ratio
                yc = 0.5 * (base[1][1] + base[0][1]) * ratio
                xn = L * cos(theta-pi/2.)
                yn = L * sin(theta-pi/2.)
                ar = self.ax.arrow(xc, yc, xn, yn,
                                   color=symb[0], length_includes_head=True, head_width=2)
                btige_arrow += [ar]

                if btige_text[i] is None:
                    btige_text[i] = self.ax.text(base[0][0] * ratio, base[0][1] * ratio,
                                                 '%s'%str(tige_name), color='r')
                else:
                    btige_text[i].set_text('%s'%str(tige_name))
                    btige_text[i].set_x(base[0][0] * ratio)
                    btige_text[i].set_y(base[0][1] * ratio)

            self.ax.set_xlim(oldxlims)
            self.ax.set_ylim(oldylims)

        if force_redraw:
            self.canvas.draw()

    ###########################################################################

    ############## GRAPHICAL HANDLING OF LAGRANGIAN MARKS #####################

    def _addmark(self):
        """Activates the addition of a mark with the GUI.

        See below the handling if click events.
        """
        global add_mark, nbclick, mark_plt, mark_text
        add_mark = True
        nbclick = 0
        self.change_button_state()


    def _suppr_all_marks(self):
        """Suppress all marks"""
        global marks
        marks = []
        reset_graph_data()
        #TODO: update the hdf5 file
#     marks_ids_hdf = h5store.get_marks_indices(hdf5file)

#     for idt in marks_ids_hdf:
#         h5store.delete_mark(hdf5file, idt)

        # Recharge les données depuis le fichier hdf
        load_hdf5_tiges(hdf5file)

        # Replot the current image
        self.plot_image(cur_image, force_clear=True)

        self.change_button_state()


    def find_mark_tige_intersection(self, mark):
        """Find whether the mark intersects a tige.

        If an intersection is found, returns True, else returns False.

        In case of intersection, the mark dictionary is completed with the intersected tige
        and the arc length of the closest tige point toward the base,
        """
        if tiges_data is None:  # no tige present
            return False

        # Retrieve the detection step from the h5 file
        detection_step = h5store.get_detection_step(hdf5file)

        if detection_step is None:
            detection_step = DETECTION_STEP_DEFAULT

        image = mark['image']
        A, B = mark["start"], mark["end"]
        mark_center = (A + B) / 2

        # Scaling factor (conversion from pixels to cm)
        scale_cmpix = h5store.get_pixelscale(hdf5file)

        # First, the tiges are sorted according to their distance to the midpoint of the mark,
        # so that tiges more likely to intersect the mark are checked first.

        tige_midpoints = []
        for tige in range(len(tiges_data.xc)):
            xc, yc = tiges_data.xc[tige, image], tiges_data.yc[tige, image]
            mezo = len(xc) // 2
            tige_midpoints.append(array([xc[mezo], yc[mezo]]))

        distances = [norm(tige_midpoint - mark_center) for tige_midpoint in tige_midpoints]
        tige_order = argsort(distances)

        # Second, tige-mark intersection is actually checked for.

        for tige in tige_order:
            xc, yc = tiges_data.xc[tige, image], tiges_data.yc[tige, image]
            tige_points = [array([x, y]) for x, y in zip(xc, yc)]
            for i, (C, D) in enumerate(zip(tige_points, tige_points[1:])):
                if any(isnan(C)) or any(isnan(D)):
                    continue
                if do_intersect(A, B, C, D):
                    mark["tige"] = tige
                    mark["intersection_index"] = i
                    # Retrieve tige ids from the h5 file
                    hdf_tiges_id = h5store.get_tiges_indices(hdf5file)
                    hdfid = hdf_tiges_id[tige]
                    smoothed_x, smoothed_y, s, _, angles, _, _, _ = \
                            load_postprocessed_data(hdf5file, hdfid)
                    s, angles = s[image], angles[image]
                    # Remove non-valid values
                    imax = flatnonzero(s.mask == False).max() + 1
                    s = s[:imax]
                    angles = angles[:imax]
                    # mark-tige intersection coordinates
                    intersection_point = array([smoothed_x[image, i], smoothed_y[image, i]])
                    base_point = array([smoothed_x[image, 0], smoothed_y[image, 0]])
                    intersection_point_from_base = intersection_point - base_point
                    averaging_zone = 80  # in pixels
                    if scale_cmpix is not None:
                        s *= scale_cmpix
                        averaging_zone = int(round(CURVATURE_AVERAGING_LENGTH / scale_cmpix))
                        intersection_point *= scale_cmpix
                        intersection_point_from_base *= scale_cmpix
                    mark["s"] = s[i]
                    mark["intersection_point"] = intersection_point
                    mark["intersection_point_from_base"] = intersection_point_from_base
                    W = int(4 * round(tiges_data.diam[cur_tige].mean() / detection_step))
                    curvatures = get_tige_curvature(angles, s,
                                                    smoothing=True, window_width=W)[0]
                    mark["curvature"] = curvatures[i]
                    averaging_points = int(round(averaging_zone / detection_step))
                    if i + averaging_points <= len(curvatures):
                        mean_curvature = mean(curvatures[i:i+averaging_points])
                        mark["mean_curvature"] = mean_curvature
                    print("""Mark {} intersects tige {} between {} and {}, at """
                          """arc length {}, curvature {} and mean curvature {}.""".format(
                              mark["name"], tiges_names[tige], C, D, mark.get("s"),
                              mark.get("curvature"), mark.get("mean_curvature")))
                    return True

        return False


    def plot_marks(self, force_redraw=False, ratio=1):
        """Plots Lagrangian marks on the canvas.

        force_redraw: If True, forces Matplotlib to redraw the figure
        ratio: Image ratio used when plotting tiges
               (for decreased image resolution)
        """
        global mark_plt, mark_text

        # Scaling factor (conversion from pixels to cm)
        scale_cmpix = h5store.get_pixelscale(hdf5file)

        times, time_units = get_time_data()[:2]
        time_unit = time_units["time unit"]

        for i, mark in enumerate(marks):
            name = mark["name"]
            tige = mark['tige']
            image = mark['image']
            time = times[image]
            start, end = mark["start"], mark["end"]

            if len(mark_plt) <= i:
                mark_plt += [None]
            if len(mark_text) <= i:
                mark_text += [None]

            color = 'blue'
            if mark_plt[i] is None:
                mark_plt[i], = self.ax.plot(
                        [start[0] * ratio, end[0] * ratio],
                        [start[1] * ratio, end[1] * ratio],
                        color=color, linestyle='-', label='mark_%i'%name, lw=1.1,
                        picker=5)
            else:
                mark_plt[i].set_data([start[0] * ratio, end[0] * ratio],
                                     [start[1] * ratio, end[1] * ratio])

            text = "%i\n"%(name)
            if time_unit == strings["image"]:
                text += strings["image"] + " %i"%time
            else:  # if there is a proper time unit
                text += "%.1f "%time + time_unit

            # Arc length of the mark along the tige
            s = mark["s"]
            # Local curvature at the mark
            curvature = mark.get("curvature")
            # Mean curvature in a zone after the mark
            mean_curvature = mark.get("mean_curvature")

            length_unit, inv_length_unit = "pix", r"pix$^{-1}$"
            if scale_cmpix is not None:
                length_unit, inv_length_unit = "cm", r"cm$^{-1}$"

            text += "\n{:.2f}{}".format(s, length_unit)
#         text += "\n{:.2e}{}".format(curvature, inv_length_unit)
            if mean_curvature is not None:
                text += "\n{:.2e}{}".format(mean_curvature, inv_length_unit)

            if mark_text[i] is None:
                mark_text[i] = self.ax.text(start[0] * ratio, start[1] * ratio,
                                            text, color=color)
            else:
                mark_text[i].set_text(text)
                mark_text[i].set_x(start[0] * ratio)
                mark_text[i].set_y(start[1] * ratio)

        if force_redraw:
            self.canvas.draw()

    ###########################################################################

    ########################## LOAD FILES #####################################
    def _open_files(self):
        """Loads a file into the GUI.

        This file can be hdf5, or a list of images for a new processing.

        With Python2, the old .pkl format is also accepted.
        """
        global base_dir_path, data_out, imgs_out, base_tiges
        global btige_plt, btige_text, dtphoto, cidkey
        global hdf5file, tiges_data

        #TODO: Bug mac version TK and wild-cards
        #OLD, ('Images', '*.jpg,*.JPG'), ('Projet', '*.pkl')
        #ftypes = [('all files', '.*'), ('Images', '*.jpg,*.JPG'), ('Projet', '*.pkl')]

        #todo  filetypes=ftypes
        files = filedialog.askopenfilenames(parent=root,
                                            title=_("Choose images to process"))

        if files != '' and len(files) > 0:

            base_dir_path = os.path.dirname(files[0]) + '/'

            #Test si c'est un fichier de traitement ou des images qui sont chargées

            if '.h5' in files[0]:  # regular hdf5 data file

                reset_graph_data()

                # On fait une RAZ toutes les données globales dans Interekt
                reset_globals_data()

                # On enregistre le chemin vers le fichier hdf5file
                hdf5file = files[0]

                # On charge le fichier
                load_hdf5_file(hdf5file)

            elif '.pkl' in files[0]:  # outdated cPickle data file

                # When run with Python 3, Interekt can not open these files.
                if not python2:

                    messagebox.showerror("""Impossible d'ouvrir le fichier""",
                                         """En Python 3, Interekt ne lit pas les fichiers"""
                                         """ pkl. Si le fichier h5 correspondant n'existe """
                                         """pas, utilisez la version Python 2 d'Interekt.""")

                    # Restore focus on the current canvas
                    self.canvas.get_tk_widget().focus_force()

                    return

                else:  # If Python 2 is used, the pkl file is converted into hdf5.
                    output_hdf5_file = base_dir_path + 'interekt_data.h5'
                    hdf5file = output_hdf5_file

                    # Does the h5 file already exist?
                    process = True
                    if os.path.exists(hdf5file):
                        # Does the user want remove the file and go on?
                        resp = messagebox.askokcancel(
                            """Un fichier de traitement interekt_data.h5 existe""",
                            """Le fichier interekt_data.h5 existe déjà, Voulez vous """
                            """continuer et supprimer le fichier existant?""")
                        if resp:
                            os.unlink(hdf5file)
                        else:
                            process = False

                    if process:
                        reset_graph_data()
                        # Launching conversion
                        convert_pkl_to_hdf(files[0], output_hdf5_file,
                                           display_message=True)
                        # Loading the newly created file
                        load_hdf5_file(hdf5file)

            else:  # hopefuly, a list of images

                # Nom du fichier
                hdf5file = base_dir_path + 'interekt_data.h5'

                # Si on ouvre des images, il faut verifier que le fichier
                # h5 n'existe pas sinon on le supprime en mettant un
                # message d'avertissement.
                process = True
                if os.path.exists(hdf5file):
                    # On demande si on vuet continuer en supprimant le fichier
                    resp = messagebox.askokcancel(
                        """Un fichier de traitement existe""",
                        """Le fichier interekt_data.h5 existe déjà voulez vous continuer?"""
                        """\nCela supprimera le fichier interekt_data.h5 existant.""")

                    if resp:
                        os.unlink(hdf5file)
                    else:
                        process = False

                if process:
                    reset_graph_data()

                    # Loading image names
                    files_to_process = [f.encode(sys.getfilesystemencoding())
                                        for f in files]

                    if self.get_photo_datetime.get():
                        # Try to sort the photos according to their date
                        try:
                            files_to_process = sorted(files_to_process,
                                                      key=h5store.get_photo_time)
                            time_sorted = True
                        except:
                            time_sorted = False

                    # Otherwise try to sort the photos according to the number
                    # in their name
                    if not self.get_photo_datetime.get() or not time_sorted:
                        try:
                            if python2:
                                files_to_process = sorted(
                                    files_to_process,
                                    key=lambda x: int(
                                        ''.join(finddigits.findall(x.split('/')[-1]))))
                            else:
                                files_to_process = sorted(
                                    files_to_process,
                                    key=lambda x: int(
                                    ''.join(finddigits.findall(x.decode().split('/')[-1]))))
                        except:
                            print("""The images have not been sorted, because no """
                                  """number was found in their names.""")

                    # Loop over the photo files for adding them to a new HDF file
                    Nimgs = len(files_to_process)
                    for i, f in enumerate(files_to_process):
                        # On affiche l'état d'avancement
                        msg1 = _("""Creating the file interekt_data.h5\nin the folder""")
                        msg2 = _("""\nConverting images...""")
                        if python2:
                            self.display_message(
                                    msg1.decode('utf8') + " %s"%base_dir_path
                                    + msg2.decode('utf8') + " (%i/%i)"%(i, Nimgs))
                        else:
                            self.display_message(msg1 + " %s"%base_dir_path + msg2
                                                 + " (%i/%i)"%(i, Nimgs))

                        # On augmente la barre de progression du GUI
                        dstep = 1/float(Nimgs) * 100.
                        #print(dstep)
                        self.prog_bar.step(dstep)
                        self.prog_bar.update_idletasks()
                        self.master.update()

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

        self.change_button_state()
        self.plot_image(0, force_clear=True)

        # Restore focus on the current canvas
        self.canvas.get_tk_widget().focus_force()


###############################################################################

########################### Language selection ################################

def set_string_variables():
    # Text strings to be translated
    global strings
    strings["organ"] = _("organ")
    strings["Organ"] = _("organ")
    strings["for organ"] = _("for organ")
    strings["image"] = _("image")
    strings["image number"] = _("image number")
    strings["photo"] = _("photo")
    strings["photo number"] = _("photo number")
    strings["time"] = _("time")
    strings["Time"] = _("Time")
    strings["min"] = _("min")
    strings["days"] = _("days")
    strings["hours"] = _("hours")
    strings["length"] = _("length")
    strings["s"] = _("curvilinear abscissa")
    strings["radius"] = _("radius")
    strings["angle"] = _("angle")
    strings["tip angle"] = _("tip angle")
    strings["degree"] = _("(deg)")
    strings["curvature"] = _("curvature")
    strings["Spatiotemporal heatmaps"] = _("Spatiotemporal heatmaps")
    strings["growth length"] = _("growth length")
    strings["residuals"] = _("residuals")
    strings["initial"] = _("initial")
    strings["Missing data"] = _("Missing data")
    strings["Unit mismatch"] = _("Unit mismatch")

    if python2:
        for key, value in strings.items():
            strings[key] = value.decode('utf8')

# Set the local directory
localedir = 'locales'

LANGUAGES = OrderedDict({
    "English":"en",
    "Esperanto":"eo",
    "Français":"fr",
})

def set_lang(lang):
    gettext_lang = gettext.translation(
        'interekt',
        localedir=os.path.normpath(
            os.path.join(os.path.dirname(__file__), localedir)
        ),
        languages=[lang],
        fallback=True,
    )
    gettext_lang.install()
    set_string_variables()


class LanguageSelector:

    def __init__(self, master):
        self.master = master
        self.tk_lang = Tk.StringVar()
        self.language_selector = Tk.Toplevel(self.master, takefocus=True)
        self.language_selector.transient(self.master)
        self.language_selector.title("Language")
        for lang, lang_code in LANGUAGES.items():
            Tk.Radiobutton(self.language_selector,
                           text=lang,
                           indicatoron=0,  # clickable text instead of circular holes
                           value=lang_code,
                           width=20,
                           padx=20,
                           variable=self.tk_lang,
                           command=self.launch,
                          ).pack(anchor=Tk.W)

    def launch(self):
        global interekt, hdf5file

        lang = self.tk_lang.get()
        set_lang(lang)

        self.language_selector.destroy()

        interekt = Interekt(self.master)

        # If a data file has been given in command line, load it now.
        if hdf5file is not None:
            load_hdf5_file(hdf5file)
            interekt.change_button_state()
            interekt.plot_image(0, force_clear=True)
            # restore focus on the current canvas
            interekt.canvas.get_tk_widget().focus_force()


###############################################################################

################################ Main windows #################################
if __name__ == '__main__':

    root = Tk.Tk()

    ls = LanguageSelector(root)

    # If a data file has been given in command line
    if len(sys.argv) > 1 and '.h5' in sys.argv[1]:
        hdf5file = sys.argv[1]

    def onclose():
        root.destroy()
        mpl.close('all')
        #sys.exit(0)

    root.protocol('WM_DELETE_WINDOW', onclose)

    root.mainloop()
