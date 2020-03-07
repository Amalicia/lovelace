from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Pillow>=7.0.0', 'scikit-learn>=0.22.1']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)