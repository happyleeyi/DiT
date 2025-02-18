# Diffusion Transformer Model
논문명 : Scalable Diffusion Models with Transformers

## Training
![image](https://github.com/user-attachments/assets/1c49ac5d-e773-4395-b457-472528630a83)

- base model로 Diffusion Transformer 사용 - 자세한 구조는 논문 참조
- 내부 attention은 scaled dot product attention 사용
- flash attention을 쓰려 했으나 안정성 문제인지 loss가 nan이 나와 사용하지 않음
- mixed precision training 사용
- ema 사용
- 나머지 설정은 LDM과 동일, VAE도 VQ-VAE로 동일하게 사용
- 위 표와 같이 다양한 사이즈가 논문에 제시되어 있지만 대부분 XL/2 adaLN-Zero로 사용함

## Sampling
- LDM과 동일

## Result (XL/2 사용, 4배 압축 VQ-VAE 사용, CelebHQ 256*256)
- Unet 기반 LDM보다 확실히 성능이 좋아진게 느껴짐
- ![image](https://github.com/user-attachments/assets/2776a765-8439-489c-b835-c20608c5abcd)
