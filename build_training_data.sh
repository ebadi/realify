git clone https://github.com/ebadi/LicensePlateGenerator.git
cd LicensePlateGenerator
git clone https://github.com/fjean/pymeanshift.git
  cd pymeanshift
  python3 ./setup.py build
  python3 ./setup.py install
  cd ..
mkdir -p output
python3 licenseplate_noborder.py
cd ..

mkdir -p license/val/
cp LicensePlateGenerator/output/* license/test/

python3 training/image.py license/val/ ~/.keras/datasets/license/val/
python3 training/image.py license/train/ ~/.keras/datasets/license/train/
python3 training/image.py license/test/ ~/.keras/datasets/license/test/
