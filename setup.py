import setuptools

# with open("README.md", "r",encoding="utf8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="cv4t",
    version="0.0.8",
    author="Wen-Hung, Chang 張文宏",
    author_email="beardad1975@nmes.tyc.edu.tw",
    description="Computer Vision wrapper library for Teenagers",
    long_description="Computer Vision wrapper library for Teenagers",
    long_description_content_type="text/markdown",
    url="https://github.com/beardad1975/cv4t",
    #packages=setuptools.find_packages(),
    platforms=["Windows"],
    python_requires=">=3.8",
    packages=['cv4t','視覺模組'],
    package_data={'cv4t': ['model_needed/*']},
    install_requires = ['opencv-contrib-python>=4.7.0.68', 'pillow >= 9.1.1', 
                    'numpy>=1.24.1', 'mss~=7.0.1','imutils==0.5.4'],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Microsoft :: Windows",
            #"Operating System :: MacOS",
            #"Operating System :: POSIX :: Linux",
            "Natural Language :: Chinese (Traditional)",
        ],
)