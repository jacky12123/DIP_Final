# DIP_Final

Implementation of DIP final project "Multiple Objects Detection in Aerial Images".

### Installation

```shell
conda create --name dip python==3.6
conda activate dip

conda install ipython
# For FCOS
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# For M2Det & yolov7
conda install pytorch==0.4.1 torchvision -c pytorch

pip install -r requirements.txt

cd cocoapi/PythonAPI
python setup.py build_ext install

cd FCOS
python setup.py build develop --no-deps

cd M2Det
sh make.sh
```

### How to run

```
python3 late_fusion.py --img data --det1 yolov7/runs/detect/dip/labels --det2 FCOS/dip/FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x/labels --det3 FCOS/dip/FCOS_imprv_R_50_FPN_1x/labels --output results --method bayesian
```

### Acknowledgments

1. https://github.com/VDIGPKU/M2Det
2. https://github.com/WongKinYiu/yolov7
3. https://github.com/tianzhi0549/FCOS
4. https://github.com/cocodataset/cocoapi
5. https://github.com/yijingru/BBAVectors-Oriented-Object-Detection
6. https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling
