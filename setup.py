#Amber
from setuptools import setup, find_packages

setup(
    name='meamt_net',
    version='1.0.0',
    description='MEaMt-Net: Multi-task Edge-aware Mamba-enhanced Transformer Network for Semi-supervised Liver Tumor Segmentation and Quantification',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/YourUsername/MEaMt-Net',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'numpy',
        'scipy',
        'einops',
        'matplotlib',
        'scikit-image',
        'pillow',
        'opencv-python',
        'pyyaml',
        'pandas',
        'tqdm',
        'nibabel',
        'simpleitk',
        'tensorboard',
        'mamba-ssm>=1.0.1',
        'nvidia-ml-py3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
