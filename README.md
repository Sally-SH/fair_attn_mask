# fair_attn_mask

## Requirements
- Python 3.7
- Pytorch 0.4+
```
pip install -r requirements.txt
```

## Data
* imSitu
  1. Download images and annotations(2014) from [coco](http://cocodataset.org/#download)
  2. Move downloaded dataset dir to data dir
  ```
  .
  ├── ...
  ├── data                              # data dir
  │   ├── of500_images_resized          # imSitu data
  │   └── ...
  └── ...
  ```

## Train
```
python train.py --save_dir <path/to/save/model> --log_dir <path/to/log>
```

## Test
```
python test.py --save_dir <path/to/saved/model>
```
