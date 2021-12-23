# -*- coding: utf-8 -*-
import os
import re
import sys
from codecs import open
from shutil import rmtree
from setuptools import find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()


filename = os.path.join(here, 'itlubber/requirements.txt')
if os.path.exists(filename):
    with open(filename, 'r') as f:
        install_requires = [r.strip() for r in f.read().split("\n") if r and not r.startswith("#")]
else:
    install_requires = []


with open(os.path.join(here, 'itlubber/__init__.py'), encoding='utf-8') as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)


setup(
    name='itlubber',
    version=__version__,
    description='public methods for itlubber',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/itlubber/itlubber',
    author='itlubber',
    author_email='1830611168@qq.com',
    python_requires='>=3.6',
    license = 'MIT License',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    keywords='credit scorecard, anomaly detection, plot templete, utils tools',
    packages=find_packages(),
    install_requires=install_requires,
    project_urls={
        'Bug Reports': 'https://github.com/itlubber/itlubber/issues',
        'Source': 'https://github.com/itlubber/itlubber/',
    },
)
