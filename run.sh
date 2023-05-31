python -m torch.distributed.launch  train.py  --config ./configs/MagneticTileSurfaceDefects-sam-vit-b-evp512.yaml 
python -m torch.distributed.launch  train.py  --config ./configs/MagneticTileSurfaceDefects-sam-vit-b-evp1024.yaml 
python -m torch.distributed.launch  train.py  --config ./configs/MagneticTileSurfaceDefects-sam-vit-b-ft512.yaml 
python -m torch.distributed.launch  train.py  --config ./configs/MagneticTileSurfaceDefects-sam-vit-b-ft1024.yaml 