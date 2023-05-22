# fair_attn_mask
CS570 Project

원래 코드에서 celeba가 돌아가도록 parser, dataloader를 약간 수정했고, 편의상 train_imsitu, test_imsitu, train_celeba, test_celeba 모두 나눠놨습니다.
CelebA 다운로드는 main branch의 Readme 확인하시면 되고 데이터셋 위치는 기존에 imsitu가 data/of500_images_resized/xxxx.jpg 이런식의 구조였던 것을
- data_imsitu/of500_images_resized/xxxx.jpg
- data_celeba/images/xxxx.jpg
이런식으로 수정했습니다.
train 함수의 masked_preds만을 이용해서도 loss를 설정해봤는데 학습은 여전히 안되네요
