# create virtual env
python3 -m venv env

#enable virtual env
source env/bin/activate

python3 -m pip install --upgrade pip setuptools

pip3 install notebook
pip3 install IPython
pip3 install matplotlib

pip3 install "tensorflow==2.5"
pip3 install pydot
pip3 install "numpy<1.17"
# pip3 install gast==0.2.2

jupyter notebook


wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/cudnn-11.2-linux-x64-v8.1.1.33.tgz

tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

OS=ubuntu1804
echo $OS
wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 

sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
sudo apt-get update

sudo apt-get install libcudnn8
sudo apt-get install libcudnn8-dev
sudo apt-get install python3-tk


ipython3 pix2pix.py




################################################ prepare training 




#pip3 install opencv-python
#pip3 install  opencv-contrib-python==4.5.3.56
#pip3 install scikit-build


git clone https://github.com/fjean/pymeanshift.git
cd pymeanshift
python3 ./setup.py build
python3  ./setup.py install


https://github.com/fjean/pymeanshift/wiki/Install





