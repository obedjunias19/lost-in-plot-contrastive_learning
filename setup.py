from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'torch>=1.7.0',
    'torchvision>=0.8.0',
    'sentence-transformers>=2.2.0',
    'transformers>=4.5.0',
    'scikit-learn>=0.24.0',
    'numpy>=1.19.0',
    'pandas>=1.2.0',
]

setup(
    name='hierarchical_film_embedding',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Hierarchical Film Embedding Model with Contrastive Learning'
)