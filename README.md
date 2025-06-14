Example usage:

python app/main.py \
    --source input/out4_.mp4 \
    --output output \
    --weights weights/yolov8x.pt \
    --trail-len 30 \
    --line 0.3 0 0.3 1 \
    --line2 0.4 0 0.4 1 \
    --min-frames 10 \
    --min-displacement 20 \
    --reverse

Use `--reverse` when the IN/OUT orientation should be swapped.

When two lines are provided, a track must cross them sequentially.
Crossing from line A to line B counts as IN, while B to A counts as OUT
(unless `--reverse` is used, which swaps the meaning). Any other
combinations are ignored, reducing false counts from people lingering on
the threshold. With a single line specified, the old displacement based
method is used.

Each track also shows its displacement vector on the video as a yellow
arrow from the first point to the current position.
