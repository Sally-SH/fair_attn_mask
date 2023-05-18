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
python train_baias.py --save_dir <path/to/save/model> --log_dir <path/to/log>
python train_baias.py --save_dir <path/to/save/model> --log_dir <path/to/log> --mask_dir <path/to/trained/bias/model>
```

## Test
```
python test_bias.py --save_dir <path/to/trained/bias/model>
```
## Option
* Mask mode
  * pixel
  * patch
  ```
  python train_baias.py --mask_mode pixel --save_dir <path/to/save/model> --log_dir <path/to/log> --mask_dir <path/to/trained/bias/model>
  python test_bias.py --mask_mode patch --save_dir <path/to/trained/bias/model>
  ```
* Mask ratio
  ```
  python train_baias.py --mask_ratio 30 --save_dir <path/to/save/model> --log_dir <path/to/log> --mask_dir <path/to/trained/bias/model>
  python test_bias.py --mask_ratio 10 --save_dir <path/to/trained/bias/model>
  ```
