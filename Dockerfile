FROM tensorflow/tensorflow:2.3.0

RUN apt update && apt -y upgrade
RUN apt install -y python3 python3-pip

# 以下はjupyterlabの拡張機能を使うための前処理(最新版のnode.jpのインストール)
RUN apt install -y nodejs npm curl
RUN npm cache clean
RUN npm install n -g
RUN n stable
RUN ln -sf /usr/local/bin/node /usr/bin/node
RUN apt purge -y nodejs npm

WORKDIR /home
COPY requirements.txt ${PWD}
RUN pip3 install -r requirements.txt

# jupyterlabの"Table of Contents"
RUN jupyter labextension install @jupyterlab/toc
# jupyterlabの"tqdm"
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
#RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
# jupyterlabの"Variable Inspector""
RUN jupyter labextension install @lckr/jupyterlab_variableinspector

# 作業ディレクトリ
WORKDIR /home/workspace
