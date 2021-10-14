from setuptools import setup

setup(name="gecko-ml",
      version="0.1",
      description="Atmospheric chemistry emulator based on the GECKO-A model.",
      author="David John Gagne, Charlie Becker, Keely Lawrence, John Schreck, Siyuan Wang, Alma Hodzic",
      author_email="dgagne@ucar.edu",
      license="MIT",
      url="https://github.com/NCAR/gecko-ml",
      packages=["geckoml", "geckoml/torch"])
