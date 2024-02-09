ARG PYTHON_VERSION=3.11.6

# build stage
FROM python:${PYTHON_VERSION}

ARG PROJECT_NAME="search"

# install PDM
RUN pip install -U pip setuptools wheel pdm

# copy files
COPY pyproject.toml pdm.lock README.md .env /root/${PROJECT_NAME}/
COPY config/ /root/${PROJECT_NAME}/config
COPY src/ /root/${PROJECT_NAME}/src

COPY main.py /root/${PROJECT_NAME}/main.py

# install dependencies and project into the local packages directory
WORKDIR /root/${PROJECT_NAME}
RUN pdm sync --prod --no-editable
