from setuptools import setup, find_packages


setup(
    name='ouludeepknee',
    version='0.1',
    author='Aleksei Tiulpin',
    author_email='aleksei.tiulpin@oulu.fi',
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE.txt',
    long_description=open('README.md').read(),
)
