import sys

from setuptools import find_packages, setup


if sys.version_info < (3, 8):
    raise RuntimeError("Friday requires Python 3.8 or newer. Run with python3, not python.")


setup(
    name="friday-agent",
    version="0.1.0",
    description="Local Azure OpenAI web agent with host tools and job sessions.",
    packages=find_packages(include=("friday", "friday.*")),
    package_data={"friday.web": ["static/*"]},
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "uvicorn",
        "websockets",
        "httpx",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "friday=friday.__main__:main",
        ],
    },
)
