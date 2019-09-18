import setuptools
def parse_requirements(filename, session=None):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tripod-ml",
    version="0.1.0.3",
    author="Multiple authors",
    author_email="tiberiu44@gmail.com",
    description="Machine Learning tool for extracting latent representations from long sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adobe/tripod",
    packages=setuptools.find_packages(),
    install_requires = parse_requirements('requirements.txt', session=False),
    classifiers=(
        "Programming Language :: Python :: 3.0",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
)
