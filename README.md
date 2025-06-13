Example usage:

python app/main.py \
    --source input/out4_.mp4 \
    --output output \
    --weights weights/yolov8x.pt \
    --trail-len 30 \
    --line 0.3 0 0.3 1 \
    --min-frames 10 \
    --min-displacement 20

The direction of crossing is determined by the dot product of the
track's overall displacement vector (from the first point to the last)
and the normal of the reference line. The line orientation therefore
defines which side is counted as "IN" and which as "OUT".

Each track also shows this displacement vector on the video as a yellow
arrow from its first point to the current position.
