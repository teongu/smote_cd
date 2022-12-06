import setuptools
import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage('smote_cd')

    return config



def setup_package():
    metadata = dict(
        name='smote_cd',
        author='Teo Nguyen, Benoit Liquet, Kerrie Mengersen, Baptiste Bougoüin',
        maintainer='Teo Nguyen',
        maintainer_email='teo.nguyen@univ-pau.fr',
        version='0.1.10',
        description='A package to perform SMOTE on compositional data.',
        long_description=open('README.rst').read().rstrip(),
        long_description_content_type='text/x-rst',
        url='https://github.com/teongu/smote_cd',
        keywords='',
        license='BSD-3-Clause',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: MacOS',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        package_dir = {"": "smote_cd"},
        packages = setuptools.find_packages(where="smote_cd"),
        platforms=['Linux', 'macOS', 'Unix', 'Windows'],
        python_requires='>=3.7',
        install_requires=[
            "setuptools>=50.3.1",
            "numpy>=1.14.5",
            "matplotlib>=3.3.4",
            "pandas>=0.23.3",
            "scipy>=1.6.0",
            "scikit-learn>=0.24.1",
        ],
    )

    distutils_commands = {
        'check',
        'clean',
        'egg_info',
        'dist_info',
        'install_egg_info',
        'rotate',
    }
    commands = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    if all(command in distutils_commands for command in commands):
        from setuptools import setup
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration
    setup(**metadata)


if __name__ == '__main__':
    setup_package()