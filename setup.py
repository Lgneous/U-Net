from setuptools import setup


setup(
    name="unet",
    description="An implementation of the U-Net architecture with Tensorflow",
    url="https://github.com/Lgneous/U-Net",
    license="MIT",
    packages=["unet"],
    install_requires=["tensorflow-gpu==2.4.0"],
    setup_requires=["setuptools-scm==3.3.3"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
