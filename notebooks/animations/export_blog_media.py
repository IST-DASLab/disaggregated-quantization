"""Package the 1080p/30 FPS renders for Hugging Face blog video embeds.

Run after rendering ODP and ODPResults. MP4 stream-copy preserves
quality; faststart moves the index ahead of the video for progressive playback.
Poster images provide a useful still when browser autoplay is disabled.
The static QADD image is generated separately by schematics/fig_qadd_blog.py.
"""

from pathlib import Path

import av

ROOT=Path(__file__).resolve().parents[2]
MEDIA=ROOT/"notebooks/animations/media"
OUTPUT=ROOT/"notebooks/blogpost/media"


def export(source,name,poster_time):
    destination=OUTPUT/f"{name}.mp4"
    with av.open(str(source)) as video, av.open(
        str(destination),"w",options={"movflags":"+faststart"}
    ) as output:
        stream=video.streams.video[0]
        assert (stream.width,stream.height)==(1920,1080)
        assert stream.base_rate==30 and stream.codec_context.name=="h264"
        target=output.add_stream_from_template(stream)
        for packet in video.demux(stream):
            if packet.dts is not None:
                packet.stream=target
                output.mux(packet)
    with av.open(str(destination)) as video:
        for frame in video.decode(video=0):
            if frame.time>=poster_time:
                frame.to_image().save(OUTPUT/f"{name}_poster.png",optimize=True)
                break
        print(f"{destination.name}: 1920x1080, 30 FPS, {video.duration/1e6:.2f}s, "
              f"{destination.stat().st_size/1e6:.2f} MB",flush=True)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True,exist_ok=True)
    export(MEDIA/"videos/odp/1080p30/ODP.mp4","odp_mechanism",5.4)
    export(MEDIA/"videos/odp_results/1080p30/ODPResults.mp4","dq_results",8)
