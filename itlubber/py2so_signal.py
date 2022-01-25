from genericpath import exists
import os
import time
import shutil
from distutils.core import setup
from Cython.Build import cythonize


starttime = time.time()
currdir = os.path.abspath('.')
build_dir = "build"
build_tmp_dir = build_dir + "/temp"
build_files = ["anomaly_detection/anormal_location.py"]


if __name__ == '__main__':
    setup(ext_modules = cythonize(build_files),script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir])

    if os.path.exists(build_tmp_dir):
        shutil.rmtree(build_tmp_dir)

    for f in [f.replace(".py", ".c") for f in build_files]:
        try:
            os.remove(f)
        except:
            pass
