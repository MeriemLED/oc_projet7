#!/bin/bash

sed -ri 's/([0-9]{1,3}\.){3}[0-9]{1,3}/'"`hostname -I`/" *.py

apt update && apt upgrade -y

apt install python3-pip -y

pip3 install -r requirements.txt

nohup python3 API.py

nohup streamlit run Dashboard.py


