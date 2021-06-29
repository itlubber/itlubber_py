from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='packaging_tutorial',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/itlubber/sampleproject',
    license='MIT',
    author='itlubber',
    author_email='1830611168@qq.com',
    description='public methods for itlubber',
    long_description=long_description,
    long_description_content_type="itlubber/methods",
    install_requires=['time', 'numpy', 'matplotlib', 'sklearn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)