**Realify**, generates realistic looking license plate from outlines.

![Realify](resources/sample.png)

### Demo

A short [video demo **realify** ](https://www.youtube.com/watch?v=D-7qlTAg3Zw)


### Installation
Create and enable **virtual env**:
```
git clone https://github.com/ebadi/realify
cd realify
python3 -m venv env
source env/bin/activate
```
Install needed packages in the virtual env:
```
python3 -m pip install --upgrade pip setuptools
pip3 install matplotlib
pip3 install "tensorflow==2.5"
pip3 install pydot
pip3 install "numpy<1.17"
#pip3 install gast==0.2.2
#pip3 install notebook
#pip3 install IPython
```

Install NVIDIA cudnn:
```
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
```

### Training
Install packages needed for training:

```
pip3 install Pillow
pip3 install opencv-python
pip3 install opencv-contrib-python==4.5.3.56
pip3 install scikit-build
```

The follwoing script automatically install [LicensePlateGenerator](https://github.com/ebadi/LicensePlateGenerator) and [pymeanshift](https://github.com/fjean/pymeanshift/wiki/Install) and build the training data:

```
./build_training_data.sh
./train_model.sh
```

Run the model:
```
./exec_model.sh
```




This work is done by [Infotiv AB](https://www.infotiv.se) under [VALU3S](https://valu3s.eu) project in a collaboration with [RISE](https://www.ri.se). This project has received funding from the [ECSEL](https://www.ecsel.eu) Joint Undertaking (JU) under grant agreement No 876852. The JU receives support from the European Union’s Horizon 2020 research and innovation programme and Austria, Czech Republic, Germany, Ireland, Italy, Portugal, Spain, Sweden, Turkey.

The ECSEL JU and the European Commission are not responsible for the content on this website or any use that may be made of the information it contains.


INFOTIV AB | RISE Research Institutes of Sweden | VALU3S Project
------------ |  ------------ | ------------ 
![](resources/logos/INFOTIV-logo.png)  | ![](resources/logos/RISE-logo.png)  |  ![](resources/logos/VALU3S-logo.png) 

[Realify](https://github.com/ebadi/ScenarioGenerator) project is started and is currently maintained by [Hamid Ebadi](https://github.com/ebadi).
