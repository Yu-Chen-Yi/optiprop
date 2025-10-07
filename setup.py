#!/usr/bin/env python3
"""
Field Propagation Library Setup Script
"""

from setuptools import setup, find_packages
import os

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Field Propagation Library - A Python library for optical field propagation simulation"

def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []

setup(
    name="optiprop",
    version="1.0.1",
    author="Yu-Chen-Yi",
    author_email="chenyi@g.ncu.edu.tw",
    description="A Python library for optical field propagation simulation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Yu-Chen-Yi/optiprop",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
        ],
    },
    keywords=[
        "optics",
        "physics",
        "simulation",
        "diffraction",
        "propagation",
        "fresnel",
        "angular spectrum",
        "rayleigh-sommerfeld",
        "pytorch",
        "gpu"
    ],
    project_urls={
        "Bug Reports": "https://github.com/Yu-Chen-Yi/optiprop/issues",
        "Source": "https://github.com/Yu-Chen-Yi/optiprop",
        "Documentation": "https://optiprop.readthedocs.io/",
    },
    include_package_data=True,
    zip_safe=False,
)
