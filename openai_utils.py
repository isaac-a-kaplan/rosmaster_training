"""
Robot Navigation Fine-Tuning
====================================

This single script offers a minimal end-to-end workflow for
1. **Preparing** a multimodal (image to command) dataset as a JSONL file compatible with OpenAI fine-tuning.
2. **Uploading** the dataset to OpenAI.
3. **Launching** a fine-tune job.
4. **Querying** the resulting model to get low-level movement commands.

---
💡 **How to use**
--------------------------------------------------------------------
$ export OPENAI_API_KEY="sk-..."           # or set via .env/.bashrc
$ python robot_navigation_finetune_template.py prepare  ./dataset  train.jsonl
$ python robot_navigation_finetune_template.py upload   train.jsonl
$ python robot_navigation_finetune_template.py finetune <file_id>  gpt-4o-mini-preview
# wait until job completes ➜ note the model name
$ python robot_navigation_finetune_template.py predict  <model_name>  path/to/test.jpg
--------------------------------------------------------------------

Replace `<model_name>` with the name returned by the `fine_tune` step
(e.g. `ft:gpt-4o-mini-preview:my-org:nav-bot:9f3ed2b1`).

Structure your input dataset directory like so:

    dataset/
      0001.jpg           0001.txt  # "move forward 20 centimeters"
      0002.jpg           0002.txt  # "turn left 45 degrees"
      ...

Each *.txt* file contains **one** command string that the robot should
execute when seeing the corresponding image.
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import List

import openai

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(path: Path) -> str:
    """Return a Base64-encoded image string suitable for JSONL."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(img_before_b64: str, img_after_b64: str, command: str) -> dict:
    """Create a chat completion training example with two images."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Current state (before) and desired state (after)."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_before_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_after_b64}"},
                    },
                ],
            },
            {"role": "assistant", "content": command.strip()},
        ]
    }


def find_pair_stems(directory: Path) -> List[str]:
    """Return base filename stems that have *_before.jpg, *_after.jpg, and *.txt."""
    before_files = {p.stem.replace("_before", "") for p in directory.glob("*_before.*")}
    after_files = {p.stem.replace("_after", "") for p in directory.glob("*_after.*")}
    txt_files   = {p.stem for p in directory.glob("*.txt")}
    return sorted(before_files & after_files & txt_files)


def build_dataset(images_dir: Path, out_jsonl: Path) -> None:
    lines: List[str] = []
    for stem in find_pair_stems(images_dir):
        before_path = images_dir / f"{stem}_before.jpg"
        after_path  = images_dir / f"{stem}_after.jpg"
        txt_path    = images_dir / f"{stem}.txt"
        if not (before_path.exists() and after_path.exists() and txt_path.exists()):
            print(f"Missing files for stem '{stem}' — skipping")
            continue
        img_b64_before = encode_image(before_path)
        img_b64_after  = encode_image(after_path)
        command = txt_path.read_text(encoding="utf-8")
        messages = build_messages(img_b64_before, img_b64_after, command)
        lines.append(json.dumps(messages, ensure_ascii=False))
    out_jsonl.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} example(s) -> {out_jsonl}")


# ---------------------------------------------------------------------------
# OpenAI interaction helpers
# ---------------------------------------------------------------------------

def upload_file(path: Path) -> str:
    """Upload a JSONL file and return its file_id."""
    resp = openai.files.create(file=open(path, "rb"), purpose="fine-tune")
    file_id = resp.id
    print(f"Uploaded {path} as file_id={file_id}")
    return file_id


def launch_finetune(file_id: str, base_model: str) -> str:
    """Kick off the fine-tuning job and return job_id."""
    resp = openai.fine_tuning.jobs.create(
        training_file=file_id,
        model=base_model,
    )
    job_id = resp.id
    print(f"Launched fine-tune job {job_id} -> based on {base_model}")
    return job_id


def wait_for_job(job_id: str, poll_seconds: int = 30) -> str:
    """Poll the job until it succeeds, then return the fine-tuned model name."""
    print("Waiting for job to complete ...")
    while True:
        job = openai.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        if status == "succeeded":
            model_name = job.fine_tuned_model
            print(f"Job complete! Model = {model_name}")
            return model_name
        elif status == "failed":
            raise RuntimeError(f"Fine-tune failed: {job}")
        time.sleep(poll_seconds)


def generate_command(model: str, image_path: Path, task: str, temperature: float = 0) -> str:
    """Send an image to the fine-tuned model along with a task 
       and get a sequence of movement commands."""

    task_prompt = """
You are an on-board navigation expert named Mikail.
Given the robot's current camera view (before) and a task to complete,
respond with a minimal sequence of low-level motion command that will complete the task.
Use the format: <action1> <value1>\n<action2> <value2>\n...\n<actionN> <valueN>.
The possible actions are: forward, backward, left, right, clockwise, counterc   lockwise.
The values are either distances in centimeters or rotation in rads.
"""
            
    image_b64 = encode_image(image_path)
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": task_prompt},
            {
                "role": "user",
                "content": [ 
                    {"type": "text", "text": task},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ],
            }
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser("Fine-tune vision model -> navigate robot (before+after+task)")
    sp = p.add_subparsers(dest="cmd", required=True)

    # prepare
    prep = sp.add_parser("prepare", help="Convert dataset folder to JSONL for fine-tuning")
    prep.add_argument("images_dir")
    prep.add_argument("jsonl_out")

    # upload
    up = sp.add_parser("upload", help="Upload JSONL to OpenAI")
    up.add_argument("jsonl")

    # fine‑tune
    ft = sp.add_parser("finetune", help="Launch fine-tuning job")
    ft.add_argument("file_id")
    ft.add_argument("base_model", nargs="?", default="gpt-4.1-2025-04-14")

    # predict
    pred = sp.add_parser("predict", help="Generate low-level command from images + task")
    pred.add_argument("model")
    pred.add_argument("img")
    pred.add_argument("task", help="High-level navigational task instruction")

    args = p.parse_args()

    if args.cmd == "prepare":
        build_dataset(Path(args.images_dir), Path(args.jsonl_out))
    elif args.cmd == "upload":
        upload_file(Path(args.jsonl))
    elif args.cmd == "finetune":
        job = launch_finetune(args.file_id, args.base_model)
        wait_for_job(job)
    elif args.cmd == "predict":
        cmd_out = generate_command(
            args.model,
            Path(args.img),
            args.task,
            temperature=0.07,
        )
        print("=>", cmd_out)

