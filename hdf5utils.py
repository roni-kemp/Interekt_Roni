# coding: utf-8

"""
Fichier contenant des fonction utiles pour fonctionner avec hdf5 et h5py

Principalement issues du projet Silx,légèrement modifiée Voir:
http://www.silx.org/doc/silx/0.7.0/modules/io/dictdump.html
"""
import h5py
import sys
import numpy as np

string_types = (basestring,) if sys.version_info[0] == 2 else (str,)

def _prepare_hdf5_dataset(array_like):
    """Cast a python object into a numpy array in a HDF5 friendly format.

    :param array_like: Input dataset in a type that can be digested by
        ``numpy.array()`` (`str`, `list`, `numpy.ndarray`…)
    :return: ``numpy.ndarray`` ready to be written as an HDF5 dataset
    """

    # simple strings
    if isinstance(array_like, string_types):
        array_like = np.string_(array_like)

    # Ensure our data is a numpy.ndarray
    if not isinstance(array_like, (np.ndarray, np.string_)):
        array = np.array(array_like)
    else:
        array = array_like

    # handle list of strings or numpy array of strings
    if not isinstance(array, np.string_):
        data_kind = array.dtype.kind
        # unicode: convert to byte strings
        # (http://docs.h5py.org/en/latest/strings.html)
        if data_kind.lower() in ["s", "u"]:
            array = np.asarray(array, dtype=np.string_)

    return array


# Class pour faire une fermeture propre du fichier hdf5 quand on
# utilise une fonction qui fait des ouverture récursive (utilisé dans
# la fonction dicttoh5)
class _SafeH5FileReadWrite(object):
    """Context manager returning a :class:`h5py.File` object.

    If this object is initialized with a file path, we open the file
    and then we close it on exiting.

    If a :class:`h5py.File` instance is provided to :meth:`__init__` rather
    than a path, we assume that the user is responsible for closing the
    file.

    This behavior is well suited for handling h5py file in a recursive
    function. The object is created in the initial call if a path is provided,
    and it is closed only at the end when all the processing is finished.
    """
    def __init__(self, h5file, mode="w"):
        """

        :param h5file:  HDF5 file path or :class:`h5py.File` instance
        :param str mode:  Can be ``"r+"`` (read/write, file must exist),
            ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail if
            exists) or ``"a"`` (read/write if exists, create otherwise).
            This parameter is ignored if ``h5file`` is a file handle.
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

            
def dicttoh5(treedict, hdf5file, h5path='/', mode="w",
             overwrite_data=True, create_dataset_args=None):
    
    """Write a nested dictionary to a HDF5 file, using keys as member names.

    If a dictionary value is a sub-dictionary, a group is created. If it is
    any other data type, it is cast into a numpy array and written as a
    :mod:`h5py` dataset. Dictionary keys must be strings and cannot contain
    the ``/`` character.

    .. note::

        This function requires `h5py <http://www.h5py.org/>`_ to be installed.

    :param treedict: Nested dictionary/tree structure with strings as keys
         and array-like objects as leafs. The ``"/"`` character is not allowed
         in keys.
    :param hdf5file: HDF5 file name or handle. If a file name is provided, the
        function opens the file in the specified mode and closes it again
        before completing.
    :param h5path: Target path in HDF5 file in which scan groups are created.
        Default is root (``"/"``)
    :param mode: Can be ``"r+"`` (read/write, file must exist),
        ``"w"`` (write, existing file is lost), ``"w-"`` (write, fail if
        exists) or ``"a"`` (read/write if exists, create otherwise).
        This parameter is ignored if ``h5file`` is a file handle.
    :param overwrite_data: If ``True``, existing groups and datasets can be
        overwritten, if ``False`` they are skipped. This parameter is only
        relevant if ``h5file_mode`` is ``"r+"`` or ``"a"``.
    :param create_dataset_args: Dictionary of args you want to pass to
        ``h5f.create_dataset``. This allows you to specify filters and
        compression parameters. Don't specify ``name`` and ``data``.

    Example::


        city_area = {
            "Europe": {
                "France": {
                    "Isère": {
                        "Grenoble": "18.44 km2"
                    },
                    "Nord": {
                        "Tourcoing": "15.19 km2"
                    },
                },
            },
        }

        create_ds_args = {'compression': "gzip",
                          'shuffle': True,
                          'fletcher32': True}

        dicttoh5(city_area, "cities.h5", h5path="/area",
                 create_dataset_args=create_ds_args)
    """
    
    if not h5path.endswith("/"):
        h5path += "/"

    
    with _SafeH5FileReadWrite(hdf5file, mode=mode) as h5f:
        for key in treedict:
            if isinstance(treedict[key], dict) and len(treedict[key]):
                # non-empty group: recurse
                dicttoh5(treedict[key], h5f, h5path + str(key),
                         overwrite_data=overwrite_data,
                         create_dataset_args=create_dataset_args)

            elif treedict[key] is None or (isinstance(treedict[key], dict) and not len(treedict[key])):
                if (h5path + str(key)) in h5f:
                    if overwrite_data is True:
                        del h5f[h5path + str(key)]
                    else:
                        logger.warning('key (%s) already exists. '
                                       'Not overwriting.' % (h5path + str(key)))
                        continue
                # Create empty group
                h5f.create_group(h5path + str(key))

            else:
                ds = _prepare_hdf5_dataset(treedict[key])
                # can't apply filters on scalars (datasets with shape == () )
                if ds.shape == () or create_dataset_args is None:
                    if h5path + str(key) in h5f:
                        if overwrite_data is True:
                            del h5f[h5path + str(key)]
                        else:
                            logger.warning('key (%s) already exists. '
                                           'Not overwriting.' % (h5path + str(key)))
                            continue

                    h5f.create_dataset(h5path + str(key), data=ds)
                else:
                    if h5path + str(key) in h5f:
                        if overwrite_data is True:
                            del h5f[h5path + str(key)]
                        else:
                            logger.warning('key (%s) already exists. '
                                           'Not overwriting.' % (h5path + str(key)))
                            continue

                    h5f.create_dataset(h5path + str(key),
                                       data=ds,
                                       **create_dataset_args)




def _name_contains_string_in_list(name, strlist):
    if strlist is None:
        return False
    for filter_str in strlist:
        if filter_str in name:
            return True
    return False


def h5todict(h5file, path="/", exclude_names=None):
    """Read a HDF5 file and return a nested dictionary with the complete file
    structure and all data.

    Example of usage::

        from silx.io.dictdump import h5todict

        # initialize dict with file header and scan header
        header94 = h5todict("oleg.dat",
                            "/94.1/instrument/specfile")
        # add positioners subdict
        header94["positioners"] = h5todict("oleg.dat",
                                           "/94.1/instrument/positioners")
        # add scan data without mca data
        header94["detector data"] = h5todict("oleg.dat",
                                             "/94.1/measurement",
                                             exclude_names="mca_")


    .. note:: This function requires `h5py <http://www.h5py.org/>`_ to be
        installed.

    .. note:: If you write a dictionary to a HDF5 file with
        :func:`dicttoh5` and then read it back with :func:`h5todict`, data
        types are not preserved. All values are cast to numpy arrays before
        being written to file, and they are read back as numpy arrays (or
        scalars). In some cases, you may find that a list of heterogeneous
        data types is converted to a numpy array of strings.

    :param h5file: File name or :class:`h5py.File` object or spech5 file or
        fabioh5 file.
    :param str path: Name of HDF5 group to use as dictionary root level,
        to read only a sub-group in the file
    :param List[str] exclude_names: Groups and datasets whose name contains
        a string in this list will be ignored. Default is None (ignore nothing)
    :return: Nested dictionary
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
