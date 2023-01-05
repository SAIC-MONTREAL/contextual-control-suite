from setuptools import find_packages
from setuptools import setup

import fnmatch
import os
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Contextual Control Suite is designed to work with Python 3.6 " \
    "and greater Please install it before proceeding."


def find_data_files(package_dir, patterns, excludes=()):
    """Recursively finds files whose names match the given shell patterns."""
    paths = set()

    def is_excluded(s):
        for exclude in excludes:
            if fnmatch.fnmatch(s, exclude):
                return True
        return False

    for directory, _, filenames in os.walk(package_dir):
        if is_excluded(directory):
            continue
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # NB: paths must be relative to the package directory.
                relative_dirpath = os.path.relpath(directory, package_dir)
                full_path = os.path.join(relative_dirpath, filename)
                if not is_excluded(full_path):
                    paths.add(full_path)
    return list(paths)


setup(
    name='contextual_control_suite',
    py_modules=['contextual_control_suite'],
    install_requires=[
        'dm-control',
        'absl-py>=0.7.0',
        'dm-env',
        'future',
        'glfw',
        'labmaze',
        'lxml',
        'mujoco >= 2.1.5',
        'numpy >= 1.9.0',
        'protobuf >= 3.15.6',
        'pyopengl >= 3.1.4',
        'pyparsing < 3.0.0',
        'requests',
        'setuptools!=50.0.0',  # https://github.com/pypa/setuptools/issues/2350
        'scipy',
        'tqdm',
    ],
    packages=find_packages(),
    package_data={
        'contextual_control_suite':
            find_data_files(
                package_dir='contextual_control_suite',
                patterns=[
                    '*.amc', '*.msh', '*.png', '*.skn', '*.stl', '*.xml',
                    '*.textproto', '*.h5'
                ],
                excludes=[
                    '*/dog_assets/extras/*',
                    '*/kinova/meshes/*',  # Exclude non-decimated meshes.
                ]),
    },
    version="1.0.0",
    description="Contextual Control Suite environments.",
    author="Sahand Rezaei-Shoshtari, Charlotte Morissette",
)
