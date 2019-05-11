FROM python:latest
ADD . /app
WORKDIR /app
RUN pip install pipenv
RUN pipenv install --system
CMD ['python3', 'stream.py']
