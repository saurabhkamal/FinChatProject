from setuptools import find_packages, setup
setup(
    name="Finance_chatbot",
    version="0.1.0",
    author="Saurabh Kamal",
    author_email="saurabh.kamal1",
    packages=find_packages(),
    install_requires=[]
)

#### This will help to install any kinds of folder as a local package. I am importing some function
#### from different modules, and for this setup.py is required.