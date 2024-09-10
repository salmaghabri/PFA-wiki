# Cuda tools on wsl 2
https://www.youtube.com/watch?v=JaHVsZa2jTc
```shell

wget   https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

```
# alphapose
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md
## manually install halpecocotools
https://github.com/MVIG-SJTU/AlphaPose/issues/1002

```
mkdir /build
cd /build && git clone https://github.com/HaoyiZhu/HalpeCOCOAPI.git
cd /build/HalpeCOCOAPI/PythonAPI && python3 setup.py build develop --user
```
# demo inference
  
```shell

sudo apt-get install libgl1-mesa-glx libgl1-mesa-dev
pip install numpy==1.22.4
pip install  Matplotlib===3.8.4
pip install  scipy===1.13.1
sudo apt install python3-tk

```


```bash
python3 scripts/demo_inference.py \\ 
--cfg configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml \\
--checkpoint pretrained_models/simple_res50_256x192.pth \\
--indir examples/demo/ \\
--save_img
```

# Run guide
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/run.md
