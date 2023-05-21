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
* CelebA
  1. To download the CelebA dataset
  ```
  bash download.sh
  '''
  2. data dir
  ```
  .
  ├── data_celeba                       # data dir
  │   ├── images             
  │   │   ├── 000001.jpg                # CelebA data (image_dir)
  │   │   └── ...
  │   └── list_attr_celeba.txt          # (annotation_dir)
  └── ...
  ```
  * CelebA attributes
  ```
  '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
  'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
  'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
  'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
  'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
  'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
  ```

## Train
```
python train_baias.py --save_dir <path/to/save/model> --log_dir <path/to/log>
python train_fair.py --save_dir <path/to/save/model> --log_dir <path/to/log> --mask_dir <path/to/trained/bias/model>
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
  python train_fair.py --mask_mode pixel --save_dir <path/to/save/model> --log_dir <path/to/log> --mask_dir <path/to/trained/bias/model>
  python test_bias.py --mask_mode patch --save_dir <path/to/trained/bias/model>
  ```
* Mask ratio
  ```
  python train_fair.py --mask_ratio 30 --save_dir <path/to/save/model> --log_dir <path/to/log> --mask_dir <path/to/trained/bias/model>
  python test_bias.py --mask_ratio 10 --save_dir <path/to/trained/bias/model>
  ```
