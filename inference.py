#!/usr/bin/env python3
from tinygrad import Tensor, TinyJit
from model import RealESRGAN_x4plus
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def save_image(arr, path):
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def inference_tiled(model_path, input_path, output_path, tile=128, bleed=10, scale=4):
    model = RealESRGAN_x4plus()
    model.load_weights(model_path)

    img = load_image(input_path)
    H, W = img.shape[:2]
    print(f"Image: {W}x{H}, Tile: {tile}, Bleed: {bleed}, Scale: {scale}")

    # to keep tile size with bleed embedded
    tile = tile - (2 * bleed)

    H_pad = ((H + tile - 1) // tile) * tile
    W_pad = ((W + tile - 1) // tile) * tile
    img_padded = np.pad(img, ((bleed, H_pad - H + bleed), (bleed, W_pad - W + bleed), (0, 0)), mode='reflect')

    output_h, output_w = H_pad * scale, W_pad * scale
    output = np.zeros((output_h, output_w, 3), dtype=np.float32)

    jit = model.get_jitted()
    # x = Tensor.rand(1, 3, tile + 2 * bleed, tile + 2 * bleed)
    # jit(x).numpy()
    # print("JIT warmup done")

    tiles = [(y_start, x_start) for y_start in range(0, H_pad, tile) for x_start in range(0, W_pad, tile)]
    for y_start, x_start in tqdm(tiles, desc="Tiles"):
            y_end = min(y_start + tile, H_pad)
            x_end = min(x_start + tile, W_pad)

            tile_h = y_end - y_start
            tile_w = x_end - x_start

            input_tile = img_padded[y_start:y_end + 2 * bleed, x_start:x_end + 2 * bleed]

            x_tensor = Tensor(input_tile.transpose(2, 0, 1)[None])
            # print(x_tensor)
            output_tile = jit(x_tensor).numpy()[0].transpose(1, 2, 0)

            output[y_start * scale:y_end * scale, x_start * scale:x_end * scale] = \
                output_tile[bleed * scale:(bleed + tile_h) * scale, bleed * scale:(bleed + tile_w) * scale]

    output = output[:H * scale, :W * scale]
    save_image(output, output_path)
    print(f"Saved {output_path} ({output.shape[1]}x{output.shape[0]})")

def inference(model_path, input_path, output_path, tile=128, bleed=10):
    if tile > 0:
        inference_tiled(model_path, input_path, output_path, tile=tile, bleed=bleed)
    else:
        model = RealESRGAN_x4plus()
        model.load_weights(model_path)
        img = load_image(input_path)
        x = Tensor(img.transpose(2, 0, 1)[None])
        y = model(x)
        out = y[0].numpy().transpose(1, 2, 0)
        save_image(out, output_path)
        print(f"Saved {output_path} ({out.shape[1]}x{out.shape[0]})")

if __name__ == "__main__":
    import sys
    import os

    DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x4plus.safetensors")

    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        tile = int(sys.argv[3]) if len(sys.argv) > 3 else 128
        inference(DEFAULT_MODEL, input_path, output_path, tile=tile)
    else:
        print(f"Usage: python inference.py <input.png> <output.png> [tile]")
        print(f"  tile: {128} (default), set to 0 for full-image inference")
