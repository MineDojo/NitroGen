<img src="assets/github_banner.gif" width="100%" />

> Fork of [MineDojo/NitroGen](https://github.com/MineDojo/NitroGen) with real-time improvements

<div align="center">
  <p>
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> |
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/NitroGen"><strong>Dataset</strong></a> |
    <a href="https://nitrogen.minedojo.org/assets/documents/nitrogen.pdf"><strong>Paper</strong></a>
  </p>
</div>

## About

NitroGen is an open foundation model for generalist gaming agents. This multi-game model takes pixel input and predicts gamepad actions.

### Fork Improvements

- **Real-time processing** — real FPS without interruptions
- **Simplified launch via Docker**
- **Improved game window search**
- **Use any window resolution**
- **Faster image processing**
- **Enhanced CUDA support**
- **Process ID support** — specify process IDs instead of just names

## Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 11 |
| Python | 3.12 |
| GPU VRAM | ~10 GB (tested on RTX 5060 Ti 16GB) |
| Game | Your own copy (not distributed) |

## Installation

```bash
git clone https://github.com/dffdeeq/NitroGen-real-time.git
cd NitroGen-real-time

# Download model from HuggingFace
huggingface-cli download nvidia/NitroGen --local-dir ./models ng.pt
```

### Model Server

#### Option 1: Docker (Recommended)

Docker handles CUDA setup automatically — no manual configuration needed.

```bash
docker compose up --build --force-recreate
# use -d flag to run in the background
```

Install dependencies for `play.py`:
```bash
pip install .[play]
```

#### Option 2: Local

Requires manual CUDA setup on your system.

```bash
pip install uv

# Choose your CUDA version (cu126, cu128, cu129, cu130)
uv sync --extra cu129
```

## Usage

Start the game, then run the agent:

```bash
python scripts/play.py --process '<game_executable_name>.exe'
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--process` | Game executable name or process ID |
| `--width` | Game capture width (default: 1920) |
| `--height` | Game capture height (default: 1080) |
| `--no-record` | Disable video recording (better performance) |
| `--no-debug-save` | Disable debug saves |
| `--actions-per-step` | Actions per inference (default: 18) |

For maximum performance:
```bash
python scripts/play.py --process '<game>.exe' --no-record --no-debug-save --actions-per-step 18
```

### Finding Process Name

1. Open Task Manager (`Ctrl+Shift+Esc`)
2. Right-click the game process → **Properties**
3. Copy the name from **General** tab (ends with `.exe`)

For processes running inside other executables (like Minecraft in `javaw.exe`), use PowerShell:

```powershell
Get-Process <process_name> | Select-Object Id, ProcessName, MainWindowTitle | Format-Table -Auto
```

## Disclaimer

This project is strictly for research purposes and is not an official NVIDIA product.
