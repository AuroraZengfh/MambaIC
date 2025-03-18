# MambaIC: State Space Models for High-Performance Learned Image Compression (CVPR 2025)

This repo is the official implementation of CVPR 2025 paper

> MambaIC: State Space Models for High-Performance Learned Image Compression
>
> Fanhu Zeng, Hao Tang, Yihua Shao, Siyu Chen, Ling Shao, Yan Wang

[![arXiv](https://img.shields.io/badge/Arxiv-2503.12461-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.12461)

**Key words: Learned image compression, State space model, Context model.**

## :newspaper: News

- **[2025.03.18]** We release camera-ready submission on [arxiv](https://arxiv.org/abs/2503.12461). :cake:
- **[2025.03.15]** Github repo is available. :candy:
- **[2025.02.27]** **MambaIC** has been accepted by **CVPR 2025**! :tada:


## :open_book: Abstract
A high-performance image compression algorithm is crucial for real-time information transmission across numerous fields. Despite rapid progress in image compression, computational inefficiency and poor redundancy modeling still pose significant bottlenecks, limiting practical applications. Inspired by the effectiveness of state space models (SSMs) in capturing long-range dependencies, we leverage SSMs to address computational inefficiency in existing methods and improve image compression from multiple perspectives. In this paper, we integrate the advantages of SSMs for better efficiency-performance trade-off and propose an enhanced image compression approach through refined context modeling, which we term MambaIC. Specifically, we explore context modeling to adaptively refine the representation of hidden states. Additionally, we introduce window-based local attention into channel-spatial entropy modeling to reduce potential spatial redundancy during compression, thereby increasing efficiency. Comprehensive qualitative and quantitative results validate the effectiveness and efficiency of our approach, particularly for high-resolution image compression.

## :house: Architecture

overall structure of MambaIC
![structure](figures/structure.png)


the proposed spatial-channel context model
![context](figures/context.png)

## Environmental Setup
### Install CompressAI
```
git clone https://github.com/InterDigitalInc/CompressAI compressai
cd compressai
pip install -U pip && pip install -e .
```

### Install Vmamba
```
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
cd kernels/selective_scan && pip install .
```
### Dataset

```
MambaIC
|-- dataset
    |-- flickr30k
        |-- train_1.jpg
        |-- train_2.jpg
        ...
    |-- Kodak
        |-- kodak_1.jpg
        |-- kodak_2.jpg
        ...
    |-- CLIC
        |-- CLIC_1.jpg
        |-- CLIC_2.jpg
        ... 
    |-- Tecnick
        |-- Tecnick_1.jpg
        |-- Tecnick_2.jpg
        ...
```


###

## Training 
```
```

## Evaluation
```
```

## Experimental Results


Results on Kodak
![results](figures/results.png)

## :blue_book: Citation
If you find this work useful, consider giving this repository a star :star: and citing :bookmark_tabs: our paper as follows:

```bibtex
@article{zeng2025mambaic,
  title={MambaIC: State Space Models for High-Performance Learned Image Compression},
  author={Zeng, Fanhu and Tang, Hao and Shao, Yihua and Chen, Siyu and Shao, Ling and Wang, Yan},
  journal={arXiv preprint arXiv:2503.12461},
  year={2025}
}
```

## Note
This is not the exact original code and is a re-implementation of our CVPR 2025 paper. But the core code and experimental results are almost the same, with slight difference.

## Acknowledgememnt

The code is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI), [Mamba](https://github.com/state-spaces/mamba), [Vmamba](https://github.com/MzeroMiko/VMamba), [MambaVision](https://github.com/NVlabs/MambaVision), [ELIC](https://github.com/VincentChandelier/ELiC-ReImplemetation), [MambaVC](https://github.com/QinSY123/2024-MambaVC). Thanks for these great works and open sourcing! 

If you find them helpful, please consider citing them as well. 
