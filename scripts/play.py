import os
import sys
import time
import json
import threading
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_client import ModelClient

import argparse


class ActionState:
    """Thread-safe container for current action predictions."""

    def __init__(self, default_action):
        self.actions = []
        self.index = 0
        self.lock = threading.Lock()
        self.default_action = default_action

    def update(self, new_actions: list):
        """Replace queue with new predictions (called by inference thread)."""
        with self.lock:
            self.actions = new_actions
            self.index = 0

    def get_next(self) -> dict:
        """Get next action, or repeat last, or return default."""
        with self.lock:
            if not self.actions:
                return self.default_action

            if self.index < len(self.actions):
                action = self.actions[self.index]
                self.index += 1
                return action
            else:
                return self.actions[-1]


parser = argparse.ArgumentParser(description="VLM Inference")
parser.add_argument("--process", type=str, default="celeste.exe", help="Game to play")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--no-record", action="store_true", help="Disable video recording for faster inference")
parser.add_argument("--no-debug-save", action="store_true", help="Disable debug PNG saving")
parser.add_argument("--actions-per-step", type=int, default=None, help="Use only first N actions per predict (receding horizon). Default: use all")
parser.add_argument("--async", dest="async_mode", action="store_true", default=True, help="Async mode: game runs in real-time, inference in background (default)")
parser.add_argument("--sync", dest="async_mode", action="store_false", help="Sync mode: game paused during inference (uses speedhack)")
parser.add_argument("--width", type=int, default=1920, help="Game capture width (default: 1920)")
parser.add_argument("--height", type=int, default=1080, help="Game capture height (default: 1080)")

args = parser.parse_args()

policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
action_downsample_ratio = policy_info["action_downsample_ratio"]

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

PATH_DEBUG = PATH_REPO / "debug"
PATH_DEBUG.mkdir(parents=True, exist_ok=True)

PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = 0.5

# Find in path_out the list of existing video files, named 0001.mp4, 0002.mp4, etc.
# If they exist, find the max number and set the next number to be max + 1
video_files = sorted(PATH_OUT.glob("*_DEBUG.mp4"))
if video_files:
    existing_numbers = [f.name.split("_")[0] for f in video_files]
    existing_numbers = [int(n) for n in existing_numbers if n.isdigit()]
    next_number = max(existing_numbers) + 1
else:
    next_number = 1

PATH_MP4_DEBUG = PATH_OUT / f"{next_number:04d}_DEBUG.mp4"
PATH_MP4_CLEAN = PATH_OUT / f"{next_number:04d}_CLEAN.mp4"
PATH_ACTIONS = PATH_OUT / f"{next_number:04d}_ACTIONS.json"

def preprocess_img(main_image):
    main_cv = cv2.cvtColor(np.array(main_image), cv2.COLOR_RGB2BGR)
    final_image = cv2.resize(main_cv, (256, 256), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

zero_action = OrderedDict(
        [ 
            ("WEST", 0),
            ("SOUTH", 0),
            ("BACK", 0),
            ("DPAD_DOWN", 0),
            ("DPAD_LEFT", 0),
            ("DPAD_RIGHT", 0),
            ("DPAD_UP", 0),
            ("GUIDE", 0),
            ("AXIS_LEFTX", np.array([0], dtype=np.long)),
            ("AXIS_LEFTY", np.array([0], dtype=np.long)),
            ("LEFT_SHOULDER", 0),
            ("LEFT_TRIGGER", np.array([0], dtype=np.long)),
            ("AXIS_RIGHTX", np.array([0], dtype=np.long)),
            ("AXIS_RIGHTY", np.array([0], dtype=np.long)),
            ("LEFT_THUMB", 0),
            ("RIGHT_THUMB", 0),
            ("RIGHT_SHOULDER", 0),
            ("RIGHT_TRIGGER", np.array([0], dtype=np.long)),
            ("START", 0),
            ("EAST", 0),
            ("NORTH", 0),
        ]
    )

TOKEN_SET = BUTTON_ACTION_TOKENS


def convert_predictions_to_actions(pred, no_menu=True):
    """Convert model predictions to gamepad actions."""
    j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]
    n = len(buttons)

    env_actions = []
    for i in range(n):
        move_action = zero_action.copy()

        xl, yl = j_left[i]
        xr, yr = j_right[i]
        move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.long)
        move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.long)
        move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.long)
        move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.long)

        button_vector = buttons[i]
        for name, value in zip(TOKEN_SET, button_vector):
            if "TRIGGER" in name:
                move_action[name] = np.array([value * 255], dtype=np.long)
            else:
                move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0

        if no_menu:
            move_action["GUIDE"] = 0
            move_action["START"] = 0
            move_action["BACK"] = 0

        env_actions.append(move_action)

    return env_actions


def inference_worker(env, policy, action_state, stop_event, preprocess_fn, no_menu):
    """Background thread for model inference."""
    while not stop_event.is_set():
        try:
            obs = env.render()
            obs = preprocess_fn(obs)
            pred = policy.predict(obs)
            env_actions = convert_predictions_to_actions(pred, no_menu=no_menu)
            action_state.update(env_actions)
        except Exception as e:
            print(f"Inference error: {e}")
            time.sleep(0.1)


def run_async_loop(env, action_state, stop_event, fps=60):
    """Main loop: apply actions at fixed rate without pausing game."""
    frame_time = 1.0 / fps
    step_count = 0

    print(f"Async mode: applying actions at {fps} FPS")

    while not stop_event.is_set():
        start = time.perf_counter()

        action = action_state.get_next()
        env.apply_action(action)

        step_count += 1
        if step_count % 60 == 0:
            print(f"Applied {step_count} actions")

        elapsed = time.perf_counter() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)


print("Model loaded, starting environment...")
if args.actions_per_step:
    print(f"Receding horizon mode: using {args.actions_per_step} actions per predict (model outputs 18)")
else:
    print("Using all actions per predict")
for i in range(3):
    print(f"{3 - i}...")
    time.sleep(1)

env = GamepadEnv(
    game=args.process,
    image_width=args.width,
    image_height=args.height,
    game_speed=1.0,
    env_fps=60,
    async_mode=True,
)

# These games requires to open a menu to initialize the controller
if args.process == "isaac-ng.exe":
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

if args.process == "Cuphead.exe":
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

env.reset()

# Only pause in sync mode (speedhack)
if not args.async_mode:
    env.pause()


# Initial call to get state (only needed for sync mode)
if not args.async_mode:
    obs, reward, terminated, truncated, info = env.step(action=zero_action)
else:
    obs = env.render()

frames = None
step_count = 0

def run_loop(debug_recorder=None, clean_recorder=None):
    global obs, step_count
    try:
        while True:
            loop_start = time.perf_counter()

            obs = preprocess_img(obs)
            if not args.no_debug_save:
                obs.save(PATH_DEBUG / f"{step_count:05d}.png")

            pred = policy.predict(obs)

            j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]

            n = len(buttons)
            assert n == len(j_left) == len(j_right), "Mismatch in action lengths"

            # Receding horizon: use only first N actions if specified
            actions_to_use = args.actions_per_step if args.actions_per_step else n

            env_actions = []

            for i in range(min(actions_to_use, n)):
                move_action = zero_action.copy()

                xl, yl = j_left[i]
                xr, yr = j_right[i]
                move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.long)
                move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.long)
                move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.long)
                move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.long)

                button_vector = buttons[i]
                assert len(button_vector) == len(TOKEN_SET), "Button vector length does not match token set length"

                for name, value in zip(TOKEN_SET, button_vector):
                    if "TRIGGER" in name:
                        move_action[name] = np.array([value * 255], dtype=np.long)
                    else:
                        move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0

                env_actions.append(move_action)

            for i, a in enumerate(env_actions):
                if NO_MENU:
                    if a["START"]:
                        print("Model predicted start, disabling this action")
                    a["GUIDE"] = 0
                    a["START"] = 0
                    a["BACK"] = 0

                for _ in range(action_downsample_ratio):
                    obs, reward, terminated, truncated, info = env.step(action=a)

                    if debug_recorder and clean_recorder:
                        obs_viz = np.array(obs).copy()
                        clean_viz = cv2.resize(obs_viz, (args.width, args.height), interpolation=cv2.INTER_AREA)
                        debug_viz = create_viz(
                            cv2.resize(obs_viz, (1280, 720), interpolation=cv2.INTER_AREA),
                            i, j_left, j_right, buttons, token_set=TOKEN_SET
                        )
                        debug_recorder.add_frame(debug_viz)
                        clean_recorder.add_frame(clean_viz)

            # Append env_actions dictionnary to JSONL file
            with open(PATH_ACTIONS, "a") as f:
                for i, a in enumerate(env_actions):
                    for k, v in a.items():
                        if isinstance(v, np.ndarray):
                            a[k] = v.tolist()
                    a["step"] = step_count
                    a["substep"] = i
                    json.dump(a, f)
                    f.write("\n")

            loop_time = time.perf_counter() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0
            print(f"Step {step_count}: {len(env_actions)} actions, loop time: {loop_time*1000:.1f}ms ({fps:.1f} FPS)")
            step_count += 1
    finally:
        env.unpause()
        env.close()

if args.async_mode:
    # Async mode: game runs in real-time, inference in background
    print("Starting async mode (no speedhack, real-time gameplay)")

    action_state = ActionState(zero_action)
    stop_event = threading.Event()

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(env, policy, action_state, stop_event, preprocess_img, NO_MENU),
        daemon=True
    )
    inference_thread.start()

    try:
        run_async_loop(env, action_state, stop_event, fps=60)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        inference_thread.join(timeout=2.0)
        env.close()
else:
    # Sync mode: original behavior with speedhack
    print("Starting sync mode (with speedhack)")
    if args.no_record:
        print("Recording disabled for faster inference")
        run_loop()
    else:
        with VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="medium") as debug_recorder:
            with VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="medium") as clean_recorder:
                run_loop(debug_recorder, clean_recorder)
