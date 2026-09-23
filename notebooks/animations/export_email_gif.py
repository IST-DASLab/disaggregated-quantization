"""Make a compact, looping email GIF from the combined Twitter video.

Run with notebooks/animations/.venv/bin/python.
The first frame is a useful still for email clients that do not animate GIFs.
"""

from pathlib import Path

import av
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "media/videos/odp_results/1080p30/ODPWithResults.mp4"
OUTPUT = HERE / "media/odp_twitter_email.gif"
WIDTH, HEIGHT, FPS = 960, 540, 12


def main():
    frames = []
    with av.open(str(SOURCE)) as video:
        for frame in video.decode(video=0):
            if frame.time + 1e-6 >= len(frames) / FPS:
                frames.append(frame.reformat(width=WIDTH, height=HEIGHT,
                                             format="rgb24").to_image())

    # One palette prevents colors from flickering between frames. Include both
    # scenes and intermediate animation states; no dithering on the flat artwork.
    samples = [frames[round(i * (len(frames) - 1) / 15)] for i in range(16)]
    sheet = Image.new("RGB", (4 * WIDTH, 4 * HEIGHT))
    for i, sample in enumerate(samples):
        sheet.paste(sample, ((i % 4) * WIDTH, (i // 4) * HEIGHT))
    palette = sheet.quantize(colors=256)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE)
               for frame in frames]
    durations = [10 * (round((i + 1) * 100 / FPS) - round(i * 100 / FPS))
                 for i in range(len(indexed))]
    poster = indexed[round(5.4 * FPS)]
    poster.save(OUTPUT, save_all=True, append_images=indexed,
                duration=[1000, *durations], loop=0, optimize=True, disposal=1)
    print(f"{OUTPUT}: {WIDTH}x{HEIGHT}, {FPS} FPS, "
          f"{OUTPUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
