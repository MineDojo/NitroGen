<img src="assets/github_banner.gif" width="100%" />

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/NitroGen"><strong>Dataset</strong></a> |
    <a href="https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf"><strong>Paper</strong></a>
  </p>
</div>


# NitroGen

NitroGen is an open foundation model for generalist gaming agents. This multi-game model takes pixel input and predicts gamepad actions.

NitroGen is trained through behavior cloning on the largest video-action gameplay dataset, assembled exclusively from internet videos. It can be adapted via post-training to unseen games.

# Installation

## Prerequisites

We **do not distribute game environments**, you must use your own copies of the games. This repository only supports running the agent on **Windows games**. You can serve the model from a Linux machine for inference, but the game ultimately has to run on Windows. We have tested on Windows 11 with Python ≥ 3.12.

## Setup

Install this repo:
```bash
git clone https://github.com/MineDojo/NitroGen.git
cd NitroGen
pip install -e .
```

Download NitroGen checkpoint from [HuggingFace](https://huggingface.co/nvidia/NitroGen):
```bash
hf download nvidia/NitroGen ng.pt
```

# Getting Started

First, start an inference server for the model:
```bash
python scripts/serve.py <path_to_ng.pt>  
```

Then, run the agent on the game of your choice:
```bash
python scripts/play.py --process '<game_executable_name>.exe'
```

The `--process` parameter must be the exact executable name of the game you want to play. You can find it by right-clicking on the game process in Windows Task Manager (Ctrl+Shift+Esc), and selecting `Properties`. The process name should be in the `General` tab and end with `.exe`.
By default, screen capture uses the game window resolution. You can override it with `--width` and `--height`:
```bash
python scripts/play.py --process '<game_executable_name>.exe' --width 1920 --height 1080
```

# Paper and Citation

If you find our work useful, please consider citing us!

```bibtex
@misc{Magne2025NitroGen,
  title        = {NitroGen: An Open Foundation Model for Generalist Gaming Agents},
  author       = {Magne, Lo{\"\i}c and Awadalla, Anas and Wang, Guanzhi and Xu, Yinzhen and Belofsky, Joshua and Hu, Fengyuan and Kim, Joohwan and Schmidt, Ludwig and Gkioxari, Georgia and Kautz, Jan and Yue, Yisong and Choi, Yejin and Zhu, Yuke and Fan, Linxi},
  year         = {2025},
  howpublished = {\url{https://nitrogen.minedojo.org/}},
}
```

**Disclaimer**: This project is strictly for research purposes and is not an official NVIDIA product.
