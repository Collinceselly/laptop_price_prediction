from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function returns a list of requirements from the requirements.txt file
    '''

    #Initialize blank requirements list
    requirements = []

    #open the file
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]


        #Removing hypen_e_dot if present in requirements
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
    name='laptop_price_prediction',
    version='0.0.1',
    author='Collince Selly',
    author_email='omondicolli@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)