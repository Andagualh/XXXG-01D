FROM bitnami/pytorch

RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install freeglut3-dev -y

ADD main.py .


COPY airetro.py .
COPY hyperparameters.py hyperparameters.py
COPY test.py test.py
COPY train.py train.py
ADD models ./models
ADD opt ./opt
ADD train ./train
ADD logs ./logs
COPY GundamW-Snes /opt/bitnami/python/lib/python3.8/site-packages/retro/data/stable/GundamW-Snes
COPY GundamW-Snes.sfc ./roms/GundamW-Snes.sfc
RUN python3 -m retro.import /app/roms
CMD ["python", "./main.py"]