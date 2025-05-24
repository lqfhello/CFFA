# Cross-Modal Fine-grained Feature Alignment for Text-Based Person Re-identification

## Requirements
- Python 3.10.14
- PyTorch 2.3.0
- torchvision 0.18.0
- cuda 11.8

## Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description). 

Download the ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN).

Download the RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `./datasets` folder as follows:
```
|-- datasets/
|   |-- CUHK-PEDES/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- ICFG-PEDES/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- RSTPReid/
|       |-- imgs
|       |-- data_captions.json
```

## Training
Download the pretrained CLIP checkpoints from [here](https://huggingface.co/openai/clip-vit-base-patch16)

Then run the script
```
sh train.sh
```

## Testing
After training, you can test your model by run:
```
python test.py
```

## Acknowledgements
Some components of this code implementation are adopted from [IRRA](https://github.com/anosorae/IRRA). We sincerely appreciate for their contributions.



