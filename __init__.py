import os
import subprocess
import sys
import numpy as np


def _dummyimport():
    import Cython


try:
    from .cythonproduct2 import fastproduct

except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/openmp /O2
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False
# cython: cpow=True
# cython: cdivision=True
# cython: cdivision_warnings=False
# cython: c_api_binop_methods=True
# cython: infer_types=True
# cython: c_string_type=bytes
# cython: c_string_encoding=default

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython
np.import_array()

ctypedef fused real:
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.size_t
    cython.Py_ssize_t
    cython.float
    cython.double
    cython.longdouble
    cython.floatcomplex
    cython.doublecomplex
    cython.longdoublecomplex
    cython.Py_hash_t
    cython.Py_UCS4

ctypedef fused real2:
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.size_t
    cython.Py_ssize_t

cpdef void fastproduct(real2[:] oldval,real[:, :] empty_array,real2[:] lenlist, real[:,:] empt):

    cdef Py_ssize_t eshape=empty_array.shape[1]
    cdef Py_ssize_t oldvallen = len(oldval)
    cdef Py_ssize_t  n,o
    for o in prange(oldvallen,nogil=True):
        for n in range(eshape):
            empty_array[..., n][o]=empt[n][(oldval[o])%(lenlist[n])]
            oldval[o]=oldval[o]//lenlist[n]

"""
    pyxfile = f"cythonproduct2.pyx"
    pyxfilesetup = f"cythonproduct2mapcompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
        """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'cythonproduct2', 'sources': ['cythonproduct2.pyx'], 'include_dirs': [\'"""
        + numpyincludefolder
        + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='cythonproduct2',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .cythonproduct2 import fastproduct

    except Exception as fe:
        sys.stderr.write(f"{fe}")
        sys.stderr.flush()


def cartesian_product(*args, outputdtype=np.uint32, dtype=np.uint32):
    r"""
    Calculate the Cartesian product of input arrays.

    Parameters:
    - *args: Variable number of input arrays.
    - outputdtype (numpy.dtype): Data type of the output array.
    - dtype (numpy.dtype): Data type used for intermediate calculations.

    Returns:
    - numpy.ndarray: Cartesian product of input arrays.
    """
    argsd = {k: {k1: v1 for k1, v1 in enumerate(v)} for k, v in enumerate(args)}
    lenlist = np.array(list(map(len, args)), dtype=dtype)
    allpossibilities = np.product(lenlist)
    empty_array = np.zeros(
        (allpossibilities, len(lenlist)), dtype=outputdtype, order="C"
    )
    oldval = np.arange(allpossibilities, dtype=dtype)
    maxlen = 0
    for x in argsd:
        alen = len(argsd[x])
        maxlen = alen if alen > maxlen else maxlen
    empt = np.zeros(len(argsd.keys()) * maxlen, dtype=outputdtype).reshape(
        (len(argsd.keys()), -1)
    )
    for k, v in argsd.items():
        for kk, vv in v.items():
            empt[k][kk] = vv
    empt = empt.copy()
    fastproduct(oldval, empty_array, lenlist, empt)
    return empty_array
