#!/bin/sh
v4l2-ctl --device=/dev/video2 --set-fmt-video=width=1920,height=1080
v4l2-ctl --device=/dev/video5 --set-fmt-video=width=1920,height=1080
v4l2-ctl --device=/dev/video8 --set-fmt-video=width=1920,height=1080
v4l2-ctl --device=/dev/video11 --set-fmt-video=width=1920,height=1080
echo "changed to high resolution"
