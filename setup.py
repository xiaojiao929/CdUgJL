from setuptools import setup, find_packages

setup(
    name='meamt_net',
    version='1.0.0',
    description='MEaMt-Net: Multi-task Edge-aware Mamba-enhanced Transformer for Semi-supervised Liver Tumor Segmentation and Quantification',
    author='Amber Xiao',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
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
    ],
    python_requires='>=3.8',
)
