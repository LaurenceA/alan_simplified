from setuptools import setup, find_packages

setup(
    name = "alan_simplified",
    version = "0.0.1",
    keywords = ("test", "xxx"),
    license = "MIT Licence",

    packages = find_packages(),
    install_requires=[
    "torch>=2.0.0",
    'functorch',
    'opt-einsum'
    ],
    extras_requires=[
    'numpy',
    'pandas',
    'scipy',
    'tqdm',
    'pytest',
    'pyyaml'
    ],
    include_package_data = True,
    platforms = "any",
)
