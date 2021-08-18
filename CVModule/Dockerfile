FROM ubuntu:18.04
COPY sources.list /etc/apt/sources.list
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install --no-install-recommends --yes python3-pip ffmpeg libsm6 libxext6
# RUN apt-get update && apt-get install --no-install-recommends --yes python3.7 python3-pip ffmpeg libsm6 libxext6
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --no-cache-dir
ADD core /codes/core
ADD server /codes/server
WORKDIR /codes/server
ENTRYPOINT ["python3", "server.py"]