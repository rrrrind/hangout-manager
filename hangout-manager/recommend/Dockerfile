FROM python:3.6
USER root

RUN apt-get update
RUN apt-get -y install locales && \
          localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

EXPOSE 5000

WORKDIR /recommend
COPY requirements.txt ${PWD}
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

COPY . /recommend
