FROM python:rc-alpine
ADD . /app
WORKDIR /app
RUN apk update && apk add pkg-config && apk add libpng libpng-dev && apk add freetype freetype-dev
RUN pip install -r requirements.txt
CMD ['python3', 'stream.py']

