# para-Kor

- 한국어로 pretrained된 [GPT3-small]((https://github.com/kiyoungkim1/LMkor)) 모델을 이용하여 한국어 Paraphrase Model을 만들었습니다.
- Paraphrase Model은 Language Style Transfer 시에 핵심적인 역할을 할 수 있습니다. 

<br>

## Installation

- `para-Kor` is based on `torch=1.7(cuda 11.0)` and `python>=3.8`

- You can install locally:

```console
git clone https://github.com/L0Z1K/para-Kor.git
pip install -r requirements.txt
```

<br>

## Usage

### Make Train Dataset

- [ParaNMT-50M](https://www.aclweb.org/anthology/P18-1042/)과 같은 Paraphrase pair dataset이 있으면 좋겠지만, 한국어 dataset은 존재하질 않습니다.
- 간단하게 kor-eng pair data에서 english sentence를 [Papago Naver](https://papago.naver.com)로 번역하여 kor-kor pair를 만들었습니다. Dataset의 Quality에 따라 Model의 performance가 달라지므로 Preprocessing이 필요합니다.
- 만든 dataset의 일부를 `example.csv`에 첨부하였습니다.

### Train

- `para-Kor` can be trained as follows:
- First, you can train model with `example.csv` simply

```console
$ python script/train.py --train
```

- You can train model with your train file

```console
$ python script/train.py --train --train_file /path/to/your/train/file
```

- My command when I train my model

```console
$ python script/train.py --gpus 2 --train --accelerator ddp --train_file /workspace/data/result.csv
```

- For details,

```console
$ python script/train.py --help
```

### Test

- `para-Kor` can be tested as follows:
- `para-Kor` will be save checkpoints per epoch and per 20000 steps.

```console
$ python script/train.py --test --model_params /path/to/your/checkpoint.ckpt
```

#### Sample Results

- A와 B의 문장이 비슷한 의미이면서 구조는 조금 다르게 나오는 걸로 보아 Paraphrasing이 꽤 성공적으로 이루어졌군요.

```
A: 안녕하세요. 저는 개발자입니다.
B: 나는 개발자이다
A: 저는 열심히 살고 있습니다.
B: 나는 열심히 일하고 있다
A: 대체 몇 시간을 기다리게 하는거니
B: 도대체 몇 시간이나 기다려요
A: 나 엄청 배고픈데 빨리 밥 먹자
B: 배고파서 빨리 밥을 먹고 싶어요
A: 이제 그만 일하고 좀 쉬고 싶은 마음이에요
B: 그만 쉬고 싶어요
```

- train loss

<img width="400" alt="image" src="https://user-images.githubusercontent.com/64528476/108294375-21922580-71d9-11eb-9fb0-10b5e941988c.png">

<br>

## TODO

- 현재는 model의 performance를 직접 경험해볼 수는 없습니다.

- 추후에 fine-tuning된 model을 이용하여 api 형식으로 배포할 생각입니다.

## License

`para-Kor` project is licensed under the terms of **the Apache License 2.0**.

Copyright 2021 Seungyun Baek. All Rights Reversed.