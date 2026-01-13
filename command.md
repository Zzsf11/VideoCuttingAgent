## Visualization
- Music Caption to Video
```
python visualize_audio_segments.py --json /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/Way_Down_We_Go/captions/captions.json --output Visulization/music_keypoint/Way_Down_We_Go_max4s.mp4 --audio /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Way_Down_We_Go.mp3 
``` 
- Wave and Mel spec Visualization
```
python visualize_audio_wav.py
```

- Render the edited results
```
Python render_video.py --shot-json '/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_dark_knight_way_down_we_go_short/shot_point_gemini_wb8ofo.json' --shot-plan '/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_dark_knight_way_down_we_go_short/shot_plan_gemini.json' --video Dataset/Video/Movie/The_Dark_Knight.mkv --audio Dataset/Audio/Way_Down_We_Go.mp3  --output output_dark_knight_way_down_we_go_short/wb8ofo.mp4 --shot-scenes video_database/Video/The_Dark_Knight/frames/shot_scenes.txt --crop-ratio "9:16"  --original-audio-volume 3.0 --detection-short-side 360
```
- Shot detection results
```
python visualize_shot_comparison.py --mode single --frame_folder video_database/Video/La_La_Land/frames --segmentation video_database/Video/La_La_Land/frames/shot_scenes.txt --output Visulization/shot_detect/La_La_Land.jpg --max_frames 500
```

## Utils
- Whole Music Caption to Short
```
python -m vca.audio.short_music /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/Way_Down_We_Go/captions_max4s/captions.json -o /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/Way_Down_We_Go/captions_max4s/captions_short.json

```
