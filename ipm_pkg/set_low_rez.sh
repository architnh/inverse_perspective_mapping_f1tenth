#!/bin/sh
v4l2-ctl --device=/dev/video2 --set-fmt-video=width=960,height=540
v4l2-ctl --device=/dev/video5 --set-fmt-video=width=960,height=540  
v4l2-ctl --device=/dev/video8 --set-fmt-video=width=960,height=540  
v4l2-ctl --device=/dev/video11 --set-fmt-video=width=960,height=540  
echo "changed to low resolution"
