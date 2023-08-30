FROM python:3.10-slim
RUN apt-get update \
    && apt-get install -y libpq-dev build-essential \
    && apt-get clean

RUN mkdir /app

WORKDIR /app

ADD . .

RUN pip install --upgrade pip && pip install --user --no-cache-dir -r /app/requirements.txt

ENTRYPOINT ["bin/startup.sh"]



