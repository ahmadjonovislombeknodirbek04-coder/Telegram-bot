# How to Anything — YouTube spec ad, rendered in code

A 17.9 s spot plus a 1.7 s end card at 1920×1080 and 60 fps. It is one continuous camera move with no cuts. Everything is drawn in Python (numpy, Pillow, OpenCV) through a real pinhole perspective camera.

One red line (#FF0000) does every job, in this order:

| t (s) | beat | the line is… |
|---|---|---|
| 0.0 – 1.0 | open | born under the play icon |
| 1.0 – 4.0 | search | the focus line of the field while "how to" is typed and suggestions drop |
| 4.0 – 6.0 | **Watch it.** | the progress bar under a playing video |
| 6.0 – 7.8 | universe | a comet that flies through thousands of videos and lights their watched-bars |
| 7.8 – 10.3 | **Learn it.** | a tutorial's progress bar, jumping chapter to chapter |
| 10.3 – 12.1 | **Make it yours.** | the record button's ring (REC) |
| 12.1 – 13.5 | | the upload bar, which ends on "Published" |
| 13.5 – 15.9 | last line | the underline of "How to Anything." |
| 15.9 – 17.9 | logo | the icon outline; the logo then loads like a video (buffer, fill, play) |
| 17.9 – 19.6 | end card | the lockup and the last line, while the camera keeps drifting |

## Run

```bash
pip install numpy pillow opencv-python-headless   # ffmpeg with libx264 must be on PATH
python -m howto_ad.render                                  # -> out/how_to_anything.mp4
python -m howto_ad.render --stills 3.5,9,15 --out out/x    # PNG stills
python -m howto_ad.render --scale 0.5 --start 6 --end 8 --out out/preview.mp4
```

## How it's built

- `core.py`: the easing functions, monotone PCHIP splines, and the camera. All camera keys move forward in z, and PCHIP never overshoots, so the camera never stops and never backs up.
- `world.py`: the layout, the camera keys, the hero planes (UI rebuilt by eye), the line's states and the morphs between them, and the universe of about 2,500 cards. Cards are cleared from the flight tube and from around each hero plane.
- `assets.py`: the Pillow painters for the thumbnails, player, search field, viewfinder, type and icon.
- `render.py`: draws each frame:
  - cards at half resolution in depth-sorted depth-of-field layers
  - hero planes warped at full resolution
  - an anti-aliased light line
  - temporal sub-samples for motion blur (324° shutter, more samples when things move fast)
  - red bloom, a small dim deep-red stage glow, vignette and grain
  - BT.709 H.264 encode through ffmpeg

The renderer uses SF Pro Display/Text when it can find it (in `howto_ad/fonts/`, `~/Library/Fonts` or `/Library/Fonts`). Otherwise it falls back to Inter / Inter Display. All titles, channels and figures are fictional.
