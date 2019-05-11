# lanewarn-ml
Machine learning component(s) of LaneWarn.

We mostly utilize yolov3, due to the increased efficiency, accuracy and tooling compared to developing/training our own model in less than 24h.

## Building

```bash
docker build -t lanewarn/lanewarn-ml .
docker run --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY lanewarn/lanewarn-ml
```
