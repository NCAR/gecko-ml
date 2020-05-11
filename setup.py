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
                "scikit-learn"]

setup(name="gecko-ml",
      version="0.1",
      description="Analyze storm mode with machine learning.",
      author="David John Gagne, Keely Lawrence Siyuan Wang, Alma Hodzic",
      author_email="dgagne@ucar.edu",
      license="MIT",
      url="https://github.com/NCAR/gecko-ml",
      packages=["geckoml"],
      install_requires=requirements)
