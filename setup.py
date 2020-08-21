from setuptools import setup


requirements = ["numpy",
                "scipy",
                "xarray",
                "pandas",
                "tensorflow>=2.1.0",
                "pyyaml",
                "tqdm",
                "netcdf4",
                "matplotlib",
                "scikit-learn",
                "dask",
                "distributed",
                "keras_self_attention"]

setup(name="gecko-ml",
      version="0.1",
      description="Atmospheric chemistry emulator based on the GECKO-A model.",
      author="David John Gagne, Charlie Becker, Keely Lawrence, Siyuan Wang, Alma Hodzic",
      author_email="dgagne@ucar.edu",
      license="MIT",
      url="https://github.com/NCAR/gecko-ml",
      packages=["geckoml"],
      install_requires=requirements)
