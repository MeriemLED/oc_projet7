#!/bin/bash

sed -ri 's/([0-9]{1,3}\.){3}[0-9]{1,3}/'"`ip --br a | grep eth0 | tr -s " " | cut -d " " -f 3 | cut -d "/" -f 1`/" *.py

apt update && apt upgrade -y

apt install python3-pip -y

pip3 install -r requirements.txt

nohup python3 API.py & nohup streamlit run Dashboard.py &

