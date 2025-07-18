# setup.py
from setuptools import setup, find_packages

setup(
    name='MiMeNet',
    version='1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'scikit-bio',
        'scipy',
        'seaborn',
        'matplotlib',
    ],
)
