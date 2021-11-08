python3 ../LicensePlateGenerator/licenseplate_noborder.py
cp -r /home/wave/repositories/LicensePlateGenerator/output license/test/
python3 image.py license/val/ ~/.keras/datasets/license/val/
python3 image.py license/train/ ~/.keras/datasets/license/train/
python3 image.py license/test/ ~/.keras/datasets/license/test/


