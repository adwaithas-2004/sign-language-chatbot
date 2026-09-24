**Live path:** replaying the 192 test videos through the app's own segmenter and model gives **94.3%** top-1. 6 videos never started a sign and 0 were split into several.

Words that start a sign in under 80% of their videos (all splits): priest (20%), fall (60%), shop (64%).

**Closer cameras:** the same test videos, cropped to what a closer camera sees (down to the given number of shoulder widths below the shoulders, and at least 1.5 above and 1.5 to each side). A desk webcam often sees only 0.5-1 below the shoulders.

| Camera sees below the shoulders | Live top-1 |
|---|---|
| full view | 94.3% |
| 1.4 | 95.8% |
| 0.9 | 94.3% |
| 0.6 | 90.6% |
| 0.3 | 75.5% |
