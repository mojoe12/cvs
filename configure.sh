# Install Python
apt-get update
TZ=UTC
apt-get install -y python3.9 python3.9-venv python3.9-distutils curl build-essential
DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
rm -rf /var/lib/apt/lists/*

curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
ln -s /usr/bin/python3.9 /usr/bin/python
ln -s /usr/local/bin/pip /usr/bin/pip
rm -rf /var/lib/apt/lists/*
apt-get install -y ffmpeg libsm6 libxext6

# Prepare environment
# you can load a virtual environment as well
pip install torch torchvision torchmetrics timm pandas scikit-learn scikit-image matplotlib
