from setuptools import setup
import yaml

env_file = "environment.yml"
with open(env_file) as env:
    env_dict = yaml.load(env, Loader=yaml.Loader)
requirements = env_dict["dependencies"][1:-1]
requirements.extend(env_dict["dependencies"][-1]["pip"][:-1])
print(requirements)
setup(name="gecko-ml",
      version="0.1",
      description="Atmospheric chemistry emulator based on the GECKO-A model.",
      author="David John Gagne, Charlie Becker, Keely Lawrence, John Schreck, Siyuan Wang, Alma Hodzic",
      author_email="dgagne@ucar.edu",
      license="MIT",
      url="https://github.com/NCAR/gecko-ml",
      install_requires=requirements,
      packages=["geckoml", "geckoml/torch"])
