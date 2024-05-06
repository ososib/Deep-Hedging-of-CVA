FROM python:3.9

WORKDIR /home/osmanoscar/DH_mars

COPY . .

RUN pip install setuptools==65.5.0  "wheel<0.40.0"
RUN pip install --upgrade AutoROM
RUN pip install gym[accept-rom-license]
RUN pip install pygame --pre
RUN pip install "stable-baselines3[extra]"
RUN pip install matplotlib
RUN pip install scipy

CMD ["python", "DH.py"]
