<img src="assets/github_banner.gif" width="100%" />

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/NitroGen"><strong>Dataset</strong></a> |
    <a href="https://arxiv.org/abs/2601.02427"><strong>Paper</strong></a>
  </p>
</div>


# NitroGen

NitroGen is an open foundation model for generalist gaming agents. This multi-game model takes pixel input and predicts gamepad actions. It is trained through behavior cloning on the largest video-action gameplay dataset, assembled exclusively from internet videos. It can be adapted via post-training to unseen games.

> [!IMPORTANT]
> NitroGen is a research project to enable training generalist agent from internet data, please refer to the [paper](https://arxiv.org/abs/2601.02427) for full details. The current model is a 500M parameter DiT that can only see the last frame. Therefore it has no ability yet to plan over long horizons, play games end-to-end, self improve over time, or play completely unseen game. It is a fast-reacting system-1 sensory model


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

# Paper and Citation

If you find our work useful, please consider citing us!

```bibtex
@misc{magne2026nitrogen,
      title={NitroGen: An Open Foundation Model for Generalist Gaming Agents}, 
      author={Loïc Magne and Anas Awadalla and Guanzhi Wang and Yinzhen Xu and Joshua Belofsky and Fengyuan Hu and Joohwan Kim and Ludwig Schmidt and Georgia Gkioxari and Jan Kautz and Yisong Yue and Yejin Choi and Yuke Zhu and Linxi "Jim" Fan},
      year={2026},
      eprint={2601.02427},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.02427}, 
}
```

**Disclaimer**: This project is strictly for research purposes and is not an official NVIDIA product.
