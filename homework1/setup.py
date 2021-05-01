from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="1.1.0",
    description="Example of ml project",
    author="Mikhail Korotkov",
    install_requires=[
        "dataclasses==0.8",
        "marshmallow-dataclass==8.4.1",
        "pyyaml==5.4.1",
        "click==7.1.2",
        "pandas==1.2.4",
        "scikit-learn==0.24.1",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.2",
        "scipy==1.6.3",
        "pytest==6.2.3",
        "Faker==8.1.2",
    ],
    license="MIT",
)
