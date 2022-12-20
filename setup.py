from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.0(DEV)'
DESCRIPTION = 'WAVI data analysis in python'
LONG_DESCRIPTION = 'A package that reads raw WAVI data and provides a analysis class structure'

# Setting up
setup(
    name="pywavi",
    version=VERSION,
    author="raceee",
    author_email="<rcpeterson@noordacom.org>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas'],
    keywords=['python', 'eeg', 'neurology', 'datascience', 'machine learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)