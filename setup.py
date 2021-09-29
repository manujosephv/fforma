import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fforma-FedericoGarza", # Replace with your own username
    version="0.0.1",
    author="Manu Joseph",
    author_email="author@example.com",
    description="Fork of the FFORMA model implementation by Federico Garza Ramírez, but with lesser dependencies and clean separation between fit and predict",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manujosephv/fforma",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
