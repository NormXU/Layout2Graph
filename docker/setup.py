# coding=utf-8
# build script for 'dvedit' - Python libdv wrapper
# change this as needed
# libdvIncludeDir = "/usr/include/libdv"

import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except Exception:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org and install it")
    sys.exit(1)

SRC_DIR = "."
MODULE_NAME = "lab"
IGNORE_FILES = ["__init__.py", "config", "data", "external", "http_server", "setup.py", "mytools", "build"]


def scan_dir(dir_path, files=None):
    if files is None:
        files = []
    for f in os.listdir(dir_path):
        if f in IGNORE_FILES:
            continue
        path = os.path.join(dir_path, f)
        if os.path.isfile(path) and f not in IGNORE_FILES and path.endswith(".py"):
            files.append(path.replace(os.path.sep, ".")[:-3].replace('..', ''))
        elif os.path.isdir(path):
            scan_dir(path, files)
    return files


def make_extension(ext_name):
    ext_path = ext_name.replace(".", os.path.sep) + ".py"
    return Extension(ext_name, [ext_path], include_dirs=["."])


def get_packages(folder, packages=None):
    if packages is None:
        packages = []
    for f in os.listdir(folder):
        if f in IGNORE_FILES:
            continue
        path = os.path.join(folder, f)
        if os.path.isdir(path) and os.path.exists('{}/__init__.py'.format(path)):
            packages.append(path)
            get_packages(path, packages)
    packages.append(folder)
    return [p.replace('/', '.').replace('..', '') for p in packages]


def clean(target_dir):
    for f in os.listdir(target_dir):
        if f in IGNORE_FILES:
            continue
        path = os.path.join(target_dir, f)
        if os.path.isfile(path) and f not in IGNORE_FILES:
            os.system("rm {}".format(path))
        elif os.path.isdir(path):
            clean(path)


def copy_so(target_dir, build_base_dir, target_base_dir):
    for f in os.listdir(target_dir):
        path = os.path.join(target_dir, f)
        if os.path.isfile(path) and path.endswith(".so"):
            new_path = path.replace(build_base_dir, target_base_dir)
            os.system("cp {} {}".format(path, new_path))
        elif os.path.isdir(path):
            copy_so(path, build_base_dir, target_base_dir)


def get_build_base_dir(src_dir):
    for f in os.listdir('build'):
        if f[:3] == 'lib':
            return 'build/{}/{}'.format(f, src_dir)


if __name__ == '__main__':
    ext_names = scan_dir(SRC_DIR)
    extensions = [make_extension(name) for name in ext_names]
    tmp_packages = get_packages(SRC_DIR)
    print(tmp_packages)
    setup(
        name=MODULE_NAME,
        packages=tmp_packages,
        ext_modules=cythonize(extensions, exclude=None, nthreads=4, quiet=False,
                              compiler_directives={'language_level': sys.version_info[0],
                                                   'always_allow_keywords': True}, ),
        cmdclass={'build_ext': build_ext},
    )
    clean(SRC_DIR)
    tmp_build_base_dir = get_build_base_dir(SRC_DIR)
    copy_so(tmp_build_base_dir, tmp_build_base_dir, SRC_DIR)
    os.system('rm -rf build')
