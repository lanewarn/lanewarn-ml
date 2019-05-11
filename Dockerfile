FROM python:rc-alpine
ADD . /app
WORKDIR /app
RUN apk update && apk add libpng && apk add freetype
RUN pip install -r requirements.txt
CMD ['python3', 'stream.py']

