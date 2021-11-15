import setuptools

# with open("README.md", "r",encoding="utf8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="cv4t",
    version="0.0.5",
    author="Wen-Hung, Chang 張文宏",
    author_email="beardad1975@nmes.tyc.edu.tw",
    description="Computer Vision wrapper for Teenagers",
    long_description="Computer Vision wrapper for Teenagers",
    long_description_content_type="text/markdown",
    url="https://github.com/beardad1975/cv4t",
    #packages=setuptools.find_packages(),
    platforms=["Windows"],
    python_requires=">=3.5",
    packages=['cv4t','視覺模組'],
    package_data={'cv4t': ['models/*']},
    install_requires = ['opencv-python==4.5.3.56', 'Pillow>=8.2.0', 
                    'numpy==1.21.2', 'mss==6.0.0','imutils==0.5.4'],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Microsoft :: Windows",
            #"Operating System :: MacOS",
            #"Operating System :: POSIX :: Linux",
            "Natural Language :: Chinese (Traditional)",
        ],
)