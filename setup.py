from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='itlubber',
    version='0.0.3',
    packages=find_packages(),
    url='https://github.com/itlubber/itlubber',
    license='MIT',
    author='itlubber',
    author_email='1830611168@qq.com',
    description='public methods for itlubber',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['time', 'numpy', 'pandas', 'matplotlib', 'sklearn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)