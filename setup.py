from setuptools import setup

setup(
    name='dcmpy',
    version='0.1.0',
    description='A discrete choice modeling package in python',
    url='https://github.com/DataforProgress/dcmpy',
    author='Nic Fishman',
    author_email='njwfish@gmail.com',
    license='BSD 2-clause',
    packages=['dcmpy'],
    install_requires=[
        'cvxpy',
        'numpy',
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)