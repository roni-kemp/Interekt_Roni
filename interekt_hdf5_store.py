# coding: utf-8

"""
Script pour tester un nouveau format de sauvegarde des données pour
RootStemExtractor.

On utilise maintenant le format hdf5 qui est python indépendant, qui
va nous permettre de stocker aussi les images dedans avec une
réduction pyramidale de la résolution (pour un affichage plus rapide)

Format des données proposé:

Les images sont enregistrer en jpeg dans un array qui contient l'image
au format binaire, c'est la solution la plus rapide est la moins
gourmande en espace.

Un groupe contenant les images
/Images/
/Images/TO/RES0       qui enregistre l'image au temps 0 et à pleine résolution
/Images/TO/RES1       qui enregistre l'image au temps 0 et à la résolution plus petite 1
/Images/TO/Timestamp  qui enregistre le temps extrait de l'image en attribut
/Images/TO/Fichier    qui enregistre le nom et l'emplacement du fichier d'origine en attribut
...

/resolutions   qui enregistres les différente résolution présentes
/scalefactors  qui enregistre les facteurs de diminution de résolution
/nombre_image  qui enregistre le nombre de d'image (sans prendre en compte les differentes rosolution)



Un groupe qui contient les données relatives à chaque tiges
/datas/tiges0/nom #son nom
/datas/tiges0/xc les coordonnees du centre; dimension (temps, abscisse curviligne)
/datas/tiges0/yc les coordonnees du centre; dimension (temps, abscisse curviligne)

/datas/tiges0/postprocessing             Pour les données issues des fits etcs
/datas/tiges0/postprocessing/beta_fit    les données du fit pour beta
/datas/tiges0/postprocessing/gamma_fit ....

"""

import h5py
import numpy as np
import datetime
import PIL
import io
import os
import sys
import skimage.io as imageio
from hdf5utils import dicttoh5, h5todict
# print(io.find_available_plugins())

COMPRESSION = "gzip"  # compression en gzip, un peu lent mais comprime
                     # bien. En lzf beaucoup plus rapide mais comprime
                     # moins



def image_to_hdf(hdf5file, image_name, image_liste_position,
                 max_pyramid_layers=0, save_resolutions=False,
                 selected_resolutions=None):
    """
    Fonction pour charger les images dans le fichier hdf5, permettant
    de stocker plusieurs résolutions. Les images sont stocker comme suit:

    Si image_liste_position = 0 et max_pyramid_layers=2
    /images/T0/RES0 (shape W, H, 3)
    /images/TO/RES1 (shape int(W/2), int(H/2), 3)
    /images/TO/RES2 (shape int(W/4), int(H/4), 3)


    Paramètres
    ----------

    hdf5file, str:
        Nom du fichier hdf5 dans lequel on veut sauvegarder les
        données, ce nom doit contenir le chemin vers le fichier.

    image_name, str:
        Nom de l'image a sauvegarder, doit contenir le chemin vers
        l'image.

    image_liste_position, int:
        Position de l'image dans la série qui a servie au traitement
        (de 0 à N images)

    max_pyramid_layers, int:
        Permet de faire une réduction de taille de l'image
        initiale. On donne le nombre de fois que l'on réduit l'image
        d'un facteur downscale fixé ici à 2.

    save_resolutions, true/false:
       Permet de sauvegarder la pyramide des résolutions dans le hdf.

    selected_resolutions, None or list:
       Permet de choisir quelle résolutions on garde dans la pyramide.
       C'est une liste qui contient la position des résolution a
       sauvegarder. Expl max_pyramid_layers=2 génère RES0/RES1/RES2,
       si on met selected_resolutions = [0,2], seul RES0 et RES2 sont
       calculées et sauvegardées.
    """

    # ouverture du fichier en mode "append"
    with h5py.File(hdf5file, 'a') as f:

        # Test si le groupes images existe, sinon on le crée
        if 'images' in f:
            hdf5imggroup = f['images']
        else:
            hdf5imggroup = f.create_group('images')

        # Test si le pas de temps existe sinon on le crée
        curpos = 'T%i'%(image_liste_position)
        # Si il existe deja on le suprime
        if curpos in hdf5imggroup:
            del  hdf5imggroup[curpos]

        hdf5img = hdf5imggroup.create_group(curpos)

        # Chargement de l'image
        img = PIL.Image.open(image_name)
        rows, cols = img.size
        # print(img)

        # On crée la pyramid des résolutions
        max_size = max((rows, cols))
        # Si max_size > 4000pixels on downscale par 4, sinon par 2
        if max_size > 4000:
            downscale_factor = 4.0
        else:
            downscale_factor = 2.0

        pyramid = [(int(rows), int(cols))]
        for i in range(max_pyramid_layers):
            pyramid += [(int(rows/((i+1)*downscale_factor)),
                         int(cols/((i+1)*downscale_factor)))]


        # On crée le type (de taille variable en 8bit) pour
        # l'enregistrer dans hdf
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))

        # On va les enregistrer
        cpt_res = 0

        if selected_resolutions is None:
            selected_resolutions = range(len(pyramid))

        for i in selected_resolutions:
            p = pyramid[i]
            # print("Enregistrement de la résolution %ix%i" % p)
            # On retaille les images sauf pour la position 0 c'est
            # celle de l'image originale
            if i == 0:
                imgr = img
            else:
                imgr = img.resize(p, resample=PIL.Image.LANCZOS)
            #print(imgr.size)

            # On la sauve au format jpg dans un fichier que l'on
            # envoie dans un array numpy
            with io.BytesIO() as fimg:
                imgr.save(fimg, format='jpeg')
                fimg.seek(0)
                imgdata = np.fromstring(fimg.read(), dtype='uint8')

            res = "RES%i"%(cpt_res)
            # On enregistre l'image
            h5d = hdf5img.create_dataset(res, (1,), dt,
                                         compression=COMPRESSION,
                                         # scaleoffset=0,
                                         #shuffle=SHUFFLE,
                                         compression_opts=6)
            h5d[0] = imgdata

            cpt_res += 1

        # Pour chaque temps, on créé plusieurs attributs:
        # Le nom complet avec le path
        hdf5img.attrs['source_file'] = image_name
        # Le nom simple
        hdf5img.attrs['name'] = os.path.basename(image_name)
        # La date de la photo
        try:
            hdf5img.attrs['datetime'] = str(get_photo_time(image_name))
        except:
            print("Pas de temps pour l'image %s" % image_name)
            hdf5img.attrs['datetime'] = ""

        if save_resolutions:
            # enregistrer la pyramide de resolution
            # 1 tableau avec les résolution
            # 1 tableau avec les facteurs de reduction
            if '/resolutions' in f:
                del f['/resolutions']

            all_res = np.array([pyramid[i] for i in selected_resolutions])
            f.create_dataset('/resolutions', all_res.shape,
                             all_res.dtype,
                             data=all_res)

            if '/scalefactors' in f:
                del f['/scalefactors']

            tmp_scalefact = [(1, 1)]
            for i in range(max_pyramid_layers):
                tmp_scalefact += [((i+1)*downscale_factor, (i+1)*downscale_factor)]

            tmp_scalefact = np.array([tmp_scalefact[i] for i in selected_resolutions])
            f.create_dataset('/scalefactors', tmp_scalefact.shape,
                             tmp_scalefact.dtype,
                             data=tmp_scalefact)

def get_number_of_images(hdf5file):
    """
    Fonction qui renvoie le nombre d'image
    """

    with h5py.File(hdf5file, 'a') as f:
        if 'images' in f:
            N = len(f['/images'].keys())
        else:
            N = 0
            print("Pas d'images dans le fichier hdf5")

    return N

def save_base_tige(hdf5file, tige_id, base_pts):
    """
    Fonction pour sauvegarder les points de base d'une tige permettant
    de définir le trait de départ du code de traitement.
    """

    with h5py.File(hdf5file, 'a') as f:
        # La tige doit etre cree
        if 'data/tige%i' % tige_id in f:
            tige_base_path = 'data/tige%i/base_points' % tige_id

            # Sauvegarde des coordonnées de la base de la tige
            if tige_base_path in f:
                del f[tige_base_path]

            tmpd = np.array(base_pts)
            f.create_dataset(tige_base_path, tmpd.shape, tmpd.dtype,
                             data=tmpd,
                             compression=COMPRESSION)
        else:
            print("Il faut créer la tige avant d'enregistrer les points de base")


def tige_to_hdf5(hdf5file, tige_id, tige_name, base_pts, tige_xc, tige_yc,
                 tige_theta, tige_diam, tige_xb1, tige_yb1, tige_xb2,
                 tige_yb2):
    """
    Fonction pour sauvegarder les données d'une tige (voir l'object
    TigesManager dans new_libgravimacro) vers le fichier HDF5.

    Paramètres
    ----------

    hdf5file, string:
        Nom du fichier hdf5 dans lequel sauvegarder les données

    tige_id, int:
        Indice de la tige

    tige_name, string:
        Si on a changer l'id ou le nom de la tige (données se trouvant
        dans l'ancien tige_id_mapper.pkl)

    base_pts, list of two points:
        Coordonnées du point de base [(x1, y1), (x2, y2)] qui a servie
        au traitement de la tige (ancien pts_base du fichier
        rootstem_data.pkl).

    tige_xc, masked array:
        Coordonnée en x du centre de la tige (dimension nombre d'image
        x nombre de pas le long de la tige/racine).

    Les autres données sont tige_XX sont du même type que tige_xc pour
    les différentes variables stocker dans TigesManager.
    """

    with h5py.File(hdf5file, 'a') as f:

        # On test si on doit créer le groupe 'data' dans le fichier hdf
        if 'data' not in f:
            f.create_group('data')

        # On test si on doit créer le groupe pour cette tige
        tige_path = 'data/tige%i' % tige_id
        if tige_path not in f:
            f.create_group(tige_path)

        # Ajout des données
        hdf5tige = f[tige_path]

        # Sauvegarde des coordonnées de la base de la tige
        if 'base_points' in hdf5tige:
            del hdf5tige['base_points']

        tmpd = np.array(base_pts)
        hdf5tige.create_dataset('base_points', tmpd.shape,
                                tmpd.dtype, data=tmpd,
                                compression=COMPRESSION)

        # Sauvegarde des données issue du traitement de la tige
        for dataname in ['xc', 'yc', 'diam', 'theta', 'xb1', 'yb1', 'xb2', 'yb2']:

            # Remove the dataset if it already exist
            if dataname in hdf5tige:
                del hdf5tige[dataname]

            # On utilise eval pour recupérer la variable par rapport à
            # son nom
            tmpd = eval('tige_%s' % dataname)

            # On sauvegarde les données dans le fichier hdf
            hdf5tige.create_dataset(dataname, tmpd.shape,
                                    tmpd.dtype,
                                    data=tmpd,
                                    chunks=True,
                                    compression=COMPRESSION)

    # Sauvegarde du nom de la tige
    save_tige_name(hdf5file, tige_id, tige_name)

def get_number_of_tiges(hdf5file):
    """
    Fonction pour avoir le nombre de tiges disponibles dans le fichier hdf5
    """

    with h5py.File(hdf5file, 'a') as f:
        if 'data' in f:
            N = len([i.startswith('tige') for i in f['/data'].keys()])
        else:
            N = 0
            print("Pas de tiges dans le fichier hdf5")

    return N

def get_tiges_indices(hdf5file):
    """
    Fonction qui retourne une liste avec les id des tiges présentent
    dans le fichier hdf5.

    Cette fonction permet de récupérer toutes les tiges par exemple
    même si leur id n'est pas continu (expl on en a supprimer une).
    """

    with h5py.File(hdf5file, 'a') as f:
        if 'data' in f:
            outid = []
            for dataname in f['/data'].keys():
                if dataname.startswith('tige'):
                    outid += [int(dataname.replace('tige', ''))]
        else:
            outid = []
            print("Pas de tiges dans le fichier hdf5")

    return outid

def delete_tige(hdf5file, tigeid):
    """
    Fonction pour supprimer une tige du fichier hdf5
    """

    with h5py.File(hdf5file, 'a') as f:
        if 'data/tige%i' % tigeid in f:
            del f['data/tige%i' % tigeid]
            print('Tige %i supprimée du fichier hdf5' % tigeid)
        else:
            print('Tige %i non présente dans le fichier hdf5' %tigeid)
            present_tiges = get_tiges_indices(hdf5file)
            if present_tiges != []:
                print('Liste des ids des tiges dans le fichier hdf5')
                print(present_tiges)

def image_to_bytesio(hdf5file, position, resolution):
    """
    Fonction qui lit le contenu d'une image pour une position et une
    resolution données et retourn un objet BytesIO contenant le
    fichier de l'image brute
    """

    with h5py.File(hdf5file, 'r') as f:
        image_path = '/images/T%i/RES%i' % (position, resolution)

        if image_path in f:
            data_out = io.BytesIO(f[image_path][0])
        else:
            print("Pas d'image %s dans le fichier hdf5" % image_path)
            data_out = None

    return data_out

def open_hdf5_image(hdf5file, position, resolution):
    """
    Fonction pour ouvrir une image provenant d'un fichier hdf5, stockée sous la forme
    /images/T{position}/RES{resolution}

    retourne l'image sous la forme d'un numpy array
    """

    imgio = image_to_bytesio(hdf5file, position, resolution)
    if imgio is not None:
        data_out = imageio.imread(imgio)
    else:
        data_out = None

    return data_out

def get_image_scale(hdf5file, resolution):
    """
    Fonction qui renvoie la facteur de mise à l'échelle pour une
    résolution donnée.
    """
    scaling = None
    with h5py.File(hdf5file, 'r') as f:
        if 'scalefactors' in f:
            try:
                scaling = f['scalefactors'][resolution]
            except Exception as e:
                print("Pas de mise à l'echlle pour cette résolution dans le fichier hdf5")
                print(e)

    return scaling


def get_images_names(hdf5file, img_pos=None):
    """
    Cette fonction retourne les noms des images enregistrées dans le
    fichier hdf5 sous la forme d'une liste.

    Paramètres:
    -----------

    img_pos, None or int:
         Si on veut avoir l'info que pour une seule image, on donne sa position.
    """

    output_names = []
    with h5py.File(hdf5file, 'r') as f:
        if img_pos is None:
            max_img = len(f['/images'].keys())
            for i in range(max_img):
                key = 'T%i'%i
                name = f['/images/'+key].attrs['name']
                try:  # if Python2
                    name = name.decode()
                except:
                    pass
                output_names += [name]
        else:
            key = 'T%i' % img_pos
            name = f['/images/'+key].attrs['name']
            try:  # if Python2
                name = name.decode()
            except:
                pass
            output_names += [name]

    return output_names


def get_images_source_path(hdf5file, img_pos=None):
    """
    Cette fonction retourne les chemines vers les fichiers originaux
    des images enregistrées dans le fichier hdf5 sous la forme d'une
    liste

    Paramètres:
    -----------

    img_pos, None or int:
         Si on veut avoir l'info que pour une seule image, on donne sa position.
    """

    output_names = []
    with h5py.File(hdf5file, 'r') as f:
        if img_pos is None:
            max_img = len(f['/images'].keys())
            for i in range(max_img):
                key = 'T%i'%i
                output_names += [f['/images/'+key].attrs['source_file']]
        else:
            key = 'T%i' % img_pos
            output_names = f['/images/'+key].attrs['source_file']

    return output_names


def get_images_datetimes(hdf5file, img_pos=None):
    """
    Cette fonction permet de récupérer les timestamp des images et de
    retourner une liste de datetime.datetime objects

    Paramètres:
    -----------

    img_pos, None or int:
         Si on veut avoir l'info que pour une seule image, on donne sa position.
    """

    output_datetime = []
    with h5py.File(hdf5file, 'r') as f:
        if img_pos is None:
            max_img = len(f['/images'].keys())
            for i in range(max_img):
                key = 'T%i'%i
                str_datetime = f['/images/'+key].attrs['datetime']
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

    return output_datetime


def create_tige_in_hdf(hdf5file, tige_id):
    """
    Fonction pour créer le groupe 'tige%i'%tige_id qui va contenir les
    données relative a une tige définie par tige_id dans le fichier hdf5
    """

    with h5py.File(hdf5file, 'a') as f:

        # On test si on doit créer le groupe 'data' dans le fichier hdf
        if 'data' not in f:
            f.create_group('data')

        # On test si on doit créer le groupe pour cette tige
        tige_path = 'data/tige%i' % tige_id
        if tige_path not in f:
            f.create_group(tige_path)
            print('Tige enregistrée dans %s' % tige_path)
        else:
            print('La tige %s existe déjà dans le fichier hdf5' % tige_path)



def save_tige_dict_to_hdf(hdf5file, tige_id, dict_data):
    """
    Fonction pour sauvegarde des données relatives à une tige enregistrées
    sous forme de dictionnaire dans le fichier hdf5. Permet de
    sauvegarder l'ancien fichier tige_id_map.pkl

    Les données sont sauvegarder dans /data/tige{tige_id}/postprocessing/...
    """

    dict_group = '/data/tige%i/postprocessing' % (tige_id)
    # Test si le groupe postprocessing exist
    with h5py.File(hdf5file, 'a') as f:
        if dict_group not in f:
            f.create_group(dict_group)

    hdf5_create_args={"compression": COMPRESSION}

    dicttoh5(treedict=dict_data, hdf5file=hdf5file, h5path=dict_group,
             mode='a', create_dataset_args=hdf5_create_args)


def save_pixelscale(hdf5file, scale_cmpix):
    """
    Fonction pour sauvegarder l'échelle pixels centimètre de l'image
    """
    if scale_cmpix is None:
        goods = None
    else:
        goods = float(scale_cmpix)

    dict_data = {'scale_cmpix': goods}
    dicttoh5(treedict=dict_data, hdf5file=hdf5file, h5path='/',
             mode='a')

def get_pixelscale(hdf5file):
    """
    Fonction pour récupérer l'échelle de l'image en cm
    """
    scaleout=None
    with h5py.File(hdf5file, 'r') as f:
        if 'scale_cmpix' in f:
            scaleout = h5todict(hdf5file, path='scale_cmpix')
            if scaleout == {}:
                scaleout = None

    return scaleout


def save_steady_state(hdf5file, steady_state, exclude_steady_state_from_time_series):
    """
    Save the steady state image, and whether it should be excluded from time series.
    """
    if steady_state is None:
        steady_state_good = None
    else:
        steady_state_good = int(steady_state)

    if exclude_steady_state_from_time_series is None:
        exclude = False
    else:
        exclude = bool(exclude_steady_state_from_time_series)

    dict_data = {'steady_state': steady_state_good, 'exclude_steady_state': exclude}
    dicttoh5(treedict=dict_data, hdf5file=hdf5file, h5path='/', mode='a')

def get_steady_state(hdf5file):
    """
    Return
    - steady_state (int): the number of the steady state image
    - exclude_steady_state (bool): whether it should be excluded from time series
    """
    steady_state = None
    exclude_steady_state = False
    with h5py.File(hdf5file, 'r') as f:
        if 'steady_state' in f:
            steady_state = h5todict(hdf5file, path='steady_state')
            if steady_state == {}:
                steady_state = None
        if 'exclude_steady_state' in f:
            exclude_steady_state = h5todict(hdf5file, path='exclude_steady_state')
            if exclude_steady_state == {}:
                exclude_steady_state = False

    return steady_state, exclude_steady_state


def save_detection_step(hdf5file, step):
    """
    Fonction pour sauvegarder le pas d'espace pour la détection des squelettes.
    """
    if step is None:
        goods = None
    else:
        goods = float(step)

    dict_data = {'detection_step': goods}
    dicttoh5(treedict=dict_data, hdf5file=hdf5file, h5path='/', mode='a')

def get_detection_step(hdf5file):
    """
    Fonction pour récupérer le pas d'espace pour la détection des squelettes.
    """
    step = None
    with h5py.File(hdf5file, 'r') as f:
        if 'detection_step' in f:
            step = h5todict(hdf5file, path='detection_step')
            if step == {}:
                step = None

    return step


def get_postprocessing(hdf5file, postprocessing_name, tige_num=None):
    """

    Fonction pour récupérer les données de post traitement stockées
    dans le fichier hdf5, retourne un dictionnaire avec ces données

    Paramètres:
    -----------

    hdf5file, str:
        Fichier hdf5 contenant les données.

    postprocessing_name, str:
        Nom du dictionnaire contenant les données que l'on veut
        récupérer ce nom doit correspondre à un group ou une variable
        dans le groupe "data/tige{tige_num}/postprocessing" fichier
        hdf5.

    tige_num, None or int:
       Donne le numéro de la tige pour laquelle on veut récupérer ces
       infos, si tige_num=None, on renvoi un dictionnaire dont les
       clefs sont les numéros des tiges et les valeurs le dictionnaire
       de postprocessing demandé.
    """

    dataout = None
    if tige_num is None:
        tiges_ids = get_tiges_indices(hdf5file)
        dataout = {}
        for i in tiges_ids:
            if is_postprocessing(hdf5file, postprocessing_name, i):
                data_location = '/data/tige%i/postprocessing/%s' % (i, postprocessing_name)
                dataout[i] = h5todict(hdf5file, path=data_location)
            else:
                dataout[i] = None

    else:
        if is_postprocessing(hdf5file, postprocessing_name, tige_num):
            data_location = '/data/tige%i/postprocessing/%s' % (tige_num, postprocessing_name)
            dataout = h5todict(hdf5file, path=data_location)

    return dataout

def is_postprocessing(hdf5file, postprocessing_name, tige_num):
    """
    Fonction pour tester si les données du postprocessing definit par
    postprocessing_name est présent dans le fichier hdf5 pour la tige
    tige_num.
    """

    is_saved = False

    with h5py.File(hdf5file, 'r') as f:
        data_location = '/data/tige%i/postprocessing/%s' % (tige_num, postprocessing_name)

        if data_location in f:
            is_saved = True

    return is_saved

def is_data(hdf5file, data_name, tige_num):
    """
    Test si une tige donnée par tige_num a bien le champ data_name
    dans le fichier hdf5 stockée sous /data/tige{tige_num}/data_name
    """

    is_saved = False
    with h5py.File(hdf5file, 'r') as f:
        dataloc = '/data/tige%i/%s' % (tige_num, data_name)
        if dataloc in f:
            is_saved = True

    return is_saved

def get_tiges_bases(hdf5file, tige_num=None):
    """
    Fonction pour récupérer les bases des tiges
    """

    if tige_num is None:
        tiges_ids = get_tiges_indices(hdf5file)

        if tiges_ids != []:
            output = []
            with h5py.File(hdf5file, 'r') as f:
                for it in tiges_ids:
                    if 'data/tige%i/base_points' % it in f:
                        numpybase = f['data/tige%i/base_points' % it][:]
                        output += [numpybase.tolist()]
        else:
            output = []
    else:
        with h5py.File(hdf5file, 'r') as f:
            if 'data/tige%i/base_points' % tige_num in f:
                numpybase = f['data/tige%i/base_points' % tige_num][:]
                output = numpybase.tolist()
            else:
                output = []

    return output

def get_tiges_names(hdf5file, tige_num=None):
    """
    Fonction pour récupérer le nom d'une tige si tige_num est un
    indice d'une tige présentes dans le fichier hdf5. Si tige_num=None
    retourne l'ensemble des noms des tiges du fichier hdf5.
    """

    if tige_num is None:
        tiges_ids = get_tiges_indices(hdf5file)
    else:
        tiges_ids = [tige_num]

    tiges_names = []

    if tiges_ids != []:
        with h5py.File(hdf5file, 'r') as f:
            for idt in tiges_ids:
                if 'data/tige%i/name' % idt in f:
                    tname = h5todict(hdf5file, path='data/tige%i/name'%idt).decode()
                    tiges_names += [tname]
                else:
                    print('La tige %i a pas de nom' % idt)
                    tiges_names += [str(idt, "utf-8")]

    # Si c'est une seule tige dont on veut le nom on fait pas de liste
    if tige_num is not None:
        tiges_names = tiges_names[-1]

    return tiges_names

def save_tige_name(hdf5file, tige_id, tige_name):
    """
    Sauvegarde du nom d'une tige dans le fichier hdf5
    """

    tige_namedict = {'name': str(tige_name)}

    dicttoh5(treedict=tige_namedict, hdf5file=hdf5file,
             h5path='/data/tige%i'%tige_id,
             mode="a")

def get_tigesmanager(hdf5file):
    """
    Fonction pour créer un objet TigesManager de new_libgravimacro à
    partir des données stockées dans le fichier hdf5
    """

    # Récupération des dimension pour créer un nouveau TigesManager
    tiges_ids = get_tiges_indices(hdf5file)
    Nimg = get_number_of_images(hdf5file)

    # Si il y a des tiges
    if len(tiges_ids) > 0:
        with h5py.File(hdf5file, 'r') as f:

            # On doit estimer la taille des abscisse curviligne
            # Mais la première tige peut être vide on va donc boucler
            # pour trouver cette dimension
            Sshape = 0
            for idtige in tiges_ids:
                if 'data/tige%i/xc' % idtige in f:
                    Sshape = f['data/tige%i/xc' % idtige].shape[1]
                    break # Pour arrêter la boucle

            # Creation du tiges manager
            if Sshape == 0:
                print('Pas de données pour les tiges')
                output = None
            else:
                print(u'Création du tableau des tiges en mémoire')
                output = TigesManager(len(tiges_ids), Nimg, Sshape)

                #Ajout des données
                for i, tid in enumerate(tiges_ids):
                    print("Chargement de la tige (gui: %i, h5: tige%i)" % (i, tid))
                    for comp in ('xc', 'yc', 'theta', 'diam', 'xb1',
                                 'yb1', 'xb2', 'yb2'):
                        # Test si il y a des données pour la tige
                        h5_data_loc = 'data/tige%i/%s'%(tid, comp)
                        if h5_data_loc in f:
                            tmp = getattr(output, comp)
                            tmp[i, :, :] = f[h5_data_loc][:,:]
                        else:
                            print("Pas de données pour la tige (gui: %i, h5: tige%i)" % (i, tid))

                output.Mask_invalid()
                print("Fin du chargement")
    else:
        output = None
        print('Pas de tiges dans le fichier hdf5')

    return output


class TigesManager():

    def __init__(self, nbtige, nbimage, size=2000):
        """
            Class pour stocker toutes les tiges dans un tableau nbimage x nbtige x size
        """

        self.size = size

        modeltab = np.ones([nbtige, nbimage, size]) * 30000

        self.diam = np.ma.masked_equal(modeltab, 30000)
        self.xc = np.ma.masked_equal(modeltab, 30000)
        self.yc = np.ma.masked_equal(modeltab, 30000)
        self.theta = np.ma.masked_equal(modeltab, 30000)
        self.xb1 = np.ma.masked_equal(modeltab, 30000)
        self.yb1 = np.ma.masked_equal(modeltab, 30000)
        self.xb2 = np.ma.masked_equal(modeltab, 30000)
        self.yb2 = np.ma.masked_equal(modeltab, 30000)
        # self.gray_level = ma.masked_equal( modeltab, 30000 )

    def Mask_invalid(self):

        self.diam = np.ma.masked_equal(self.diam, 30000)
        self.xc = np.ma.masked_equal(self.xc, 30000)
        self.yc = np.ma.masked_equal(self.yc, 30000)
        self.theta = np.ma.masked_equal(self.theta, 30000)
        self.xb1 = np.ma.masked_equal(self.xb1, 30000)
        self.yb1 = np.ma.masked_equal(self.yb1, 30000)
        self.xb2 = np.ma.masked_equal(self.xb2, 30000)
        self.yb2 = np.ma.masked_equal(self.yb2, 30000)

    def compress_data(self):
        """
            Function to reduce array dimention to maximum point extracted in images
        """

        # Find the maximum of unmasked data using the first non-null sum on xc data
        try:
            iend_data = np.flatnonzero(self.xc.sum(axis=(0, 1)).mask == True)[0]
        except:
            iend_data = 0

        if iend_data > 0:
            # Loop over data to reduce them
            self.diam = self.diam[:, :, :iend_data]
            self.xc = self.xc[:, :, :iend_data]
            self.yc = self.yc[:, :, :iend_data]
            self.theta = self.theta[:, :, :iend_data]
            self.xb1 = self.xb1[:, :, :iend_data]
            self.xb2 = self.xb2[:, :, :iend_data]
            self.yb1 = self.yb1[:, :, :iend_data]
            self.yb2 = self.yb2[:, :, :iend_data]

def get_photo_time(image_path):
    """
    Fonction pour obtenir le temps de la prise de vue d'une photo a
    partir des données EXIFs
    """

    stat = PIL.Image.open(image_path)._getexif()
    ymd = stat[306].split(' ')[0].split(':')
    hms = stat[306].split(' ')[1].split(':')
    t = datetime.datetime(int(ymd[0]), int(ymd[1]), int(ymd[2]), int(hms[0]), int(hms[1]), int(hms[2]))
    # print str(stat[306])
    return t

if __name__ == '__main__':

    from pylab import *
    from new_libgravimacro import load_results

    test_data = "/home/hugo/Documents/Boulo/test_mutants_arabido/02-06-16/rootstem_data.pkl"
    rtst = load_results(test_data)
    # print(rtst.keys())
    tiges_info = rtst['tiges_info']
    tiges_data = rtst['tiges_data']
    tiges_base = rtst['pts_base']
    # print(tiges_base)
    hdf5name = '/home/hugo/Documents/Boulo/test_mutants_arabido/02-06-16/rootstem_data.h5'

    def test_load_image():
        aa = open_hdf5_image(hdf5name, 0, 2)
        if aa is not None:
            imshow(aa)
            show()

    def test_pictures_to_hdf5():
        for t in tiges_info[:10]:
            print(t['iimg'])
            # On regarde si l'image est toujours dans son dossier d'origine,
            # sinon on assume que le fichier rootstem_data.pkl est dans le
            # même dossier que les images
            if os.path.exists(t['imgname']):
                img_path = t['imgname']
            else:
                img_path = './'+os.path.basename(t['imgname'])

            image_to_hdf(hdf5name, img_path, t['iimg'], max_pyramid_layers=2)


    def test_save_tiges_data_to_hdf5():
        tige_id = 0
        tige_to_hdf5(hdf5name, tige_id, "", tiges_base[tige_id],
                     tiges_data.xc[tige_id], tiges_data.yc[tige_id],
                     tiges_data.theta[tige_id], tiges_data.diam[tige_id],
                     tiges_data.xb1[tige_id], tiges_data.yb1[tige_id],
                     tiges_data.xb2[tige_id], tiges_data.yb2[tige_id])


        test_gamma_data={'gamma_data': {'R2': 0.8878308084390526, 'RER':
                                        -0.0007212398482432093, 'gamma': 3.44221141480115, 'gamma_tilde':
                                        -4772.630662581474, 'ideb': 1, 'ideb_croissance': 9, 'ifin': 10,
                                        'ifin_croissance': 59}}

        save_tige_dict_to_hdf(hdf5name, tige_id, test_gamma_data)




    def test_get_names():
        tmp = get_images_names(hdf5name)
        print(tmp)

    def test_get_datetime():
        print(get_images_datetimes(hdf5name))

    def test_get_sourcefiles():
        print(get_images_source_path(hdf5name))

    ##########################################################################
    #test_load_image()
    test_get_names()
    test_get_datetime()
    test_get_sourcefiles()

