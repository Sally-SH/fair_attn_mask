# fair_attn_mask

## Requirements
- Python 3.7
- Pytorch 1.12+
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
* Bias-only
  ```
  python train_baias.py --save_dir <path/to/save/model>
  ```
* Fair
  ```
  python train_fair.py --ouput_dir <path/to/save/model> --mask_dir <path/to/trained/bias/model>
  ```
* Vanilla
  ```
  python main.py --output_dir <path/to/save/model>
  ```

## Test
* Bias-only
  ```
  python test_bias.py --save_dir <path/to/trained/bias/model>
  ```
## Option
* Mask mode
  * pixel
  * patch
  ```
  python train_fair.py --mask_mode pixel --output_dir <path/to/save/model> --mask_dir <path/to/trained/bias/model>
  python test_bias.py --mask_mode patch --save_dir <path/to/trained/bias/model>
  ```
* Mask ratio
  ```
  python train_fair.py --mask_ratio 30 --output_dir <path/to/save/model> --mask_dir <path/to/trained/bias/model>
  python test_bias.py --mask_ratio 10 --save_dir <path/to/trained/bias/model>
  ```
  
 ## Leakage
 * Dataset leakage $\lambda_D(a)$
   ```
   python natural_leakage.py --saved_dir <path/to/checkpoint>
   ```
 * Model leakage $\lambda_M(a)$
   ```
   python model_leakage.py --saved_dir <path/to/checkpoint>
   ```
 
