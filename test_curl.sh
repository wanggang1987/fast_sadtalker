curl -X POST -F "audio_file=@/home/redhat/AiModels/SadTalker/examples/driven_audio/RD_Radio31_000.wav" -F "image_file=@/home/redhat/AiModels/SadTalker-dev/yaqing.jpg" http://localhost:8009/create_video -o output_yq_test1.mp4


curl -X POST  -F "preprocess=\"full\"" -F "audio_file=@/home/redhat/AiModels/SadTalker/examples/driven_audio/RD_Radio31_000.wav" -F "image_file=@/home/redhat/AiModels/SadTalker-dev/yaqing.jpg" http://localhost:8009/create_video -o output_yq_test1.mp4

curl -X POST  -F "preprocess=full" -F "audio_file=@/home/redhat/AiModels/SadTalker/examples/driven_audio/RD_Radio31_000.wav" -F "image_file=@/home/redhat/AiModels/SadTalker-dev/yaqing.jpg" http://localhost:8009/create_video -o output_yq_test1.mp4

# curl -X POST -F "audio_file=@/home/redhat/AiModels/SadTalker/examples/driven_audio/RD_Radio31_000.wav" -F "image_file=@/home/redhat/AiModels/SadTalker/examples/source_image/art_5.png" http://localhost:8009/create_video -o output_test.mp4
