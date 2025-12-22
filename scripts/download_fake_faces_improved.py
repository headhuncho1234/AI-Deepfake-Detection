#!/usr/bin/env python3
"""Robust downloader for synthetic faces from thispersondoesnotexist.com.

Usage:
  python scripts/download_fake_faces_improved.py --train 250 --val 50

The script will download images with retries, validate them with Pillow,
and save them into `data/seed/fake/` and `data/validation/fake/`.
"""
import argparse
import os
import time
import requests
from PIL import Image
from io import BytesIO

URL = "https://thispersondoesnotexist.com/image"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fetch_image(session, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)" 
                      "Chrome/91.0.4472.114 Safari/537.36"
    }
    resp = session.get(URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.content

def is_valid_image(data):
    try:
        img = Image.open(BytesIO(data))
        img.verify()
        return True
    except Exception:
        return False

def save_image(data, path):
    with open(path, "wb") as f:
        f.write(data)

def download(target_count, out_dir, start_index=0, max_attempts=10, backoff=1.0):
    ensure_dir(out_dir)
    session = requests.Session()
    saved = 0
    attempts = 0
    idx = start_index
    while saved < target_count and attempts < max_attempts * target_count:
        try:
            data = fetch_image(session)
            if not is_valid_image(data):
                attempts += 1
                time.sleep(backoff)
                continue
            fname = f"fake_{idx:06d}.jpg"
            path = os.path.join(out_dir, fname)
            save_image(data, path)
            saved += 1
            idx += 1
            attempts = 0
            if saved % 10 == 0:
                print(f"Saved {saved}/{target_count} -> {path}")
            time.sleep(0.15)  # modest rate limit
        except requests.RequestException as e:
            attempts += 1
            wait = backoff * attempts
            print(f"Request failed ({e}); backing off {wait:.1f}s")
            time.sleep(wait)
    return saved

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=200, help="Number of training fake images to download")
    p.add_argument("--val", type=int, default=50, help="Number of validation fake images to download")
    p.add_argument("--seed-dir", default="data/seed/fake", help="Output directory for training fakes")
    p.add_argument("--val-dir", default="data/validation/fake", help="Output directory for validation fakes")
    args = p.parse_args()

    print("Starting fake-image downloads")
    ensure_dir(args.seed_dir)
    ensure_dir(args.val_dir)

    # Clear previous files in target dirs (ask user caution)
    for d in (args.seed_dir, args.val_dir):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    os.remove(os.path.join(d, f))
                except Exception:
                    pass

    saved_train = download(args.train, args.seed_dir, start_index=0)
    print(f"Finished training images: saved {saved_train}/{args.train}")
    saved_val = download(args.val, args.val_dir, start_index=saved_train)
    print(f"Finished validation images: saved {saved_val}/{args.val}")

    print("All done. Verify images in:")
    print("  ", args.seed_dir)
    print("  ", args.val_dir)

if __name__ == '__main__':
    main()
