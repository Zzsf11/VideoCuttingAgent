#!/bin/bash

python local_run.py \
    --Video_Path "Dataset/Video/Movie/The_Dark_Knight.mkv" \
    --Audio_Path "Dataset/Audio/Way_Down_We_Go.mp3" \
    --Instruction "A mesmerizing showcase of the Joker's chaotic presence, capturing his unsettling mannerisms—the constant licking of scarred lips, the unpredictable head tilts, the disturbingly calm smile amidst carnage. Sync his erratic movements and explosive laughter to the song's building tension, highlighting close-ups of his smeared makeup, the purple coat swirling as he orchestrates destruction, and those haunting green eyes that embody pure anarchy." \
    --instruction_type object \
    --config.MAIN_CHARACTER_NAME "Joker" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 2 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 00:43.5 -e 01:06.5

python local_run.py \
    --Video_Path "Dataset/Video/Movie/The_Dark_Knight.mkv" \
    --Audio_Path "Dataset/Audio/Way_Down_We_Go.mp3" \
    --Instruction "A psychological deconstruction of the 'White Knight's' corruption, framed through the Joker's thesis that 'madness is like gravity.' Interweave the hopeful, well-lit courtroom scenes of Harvey Dent with the charred, distorted reality of Two-Face, utilizing the Joker's interrogation monologue as the narrative spine to illustrate how the brightest hope of Gotham was systematically broken until he became the very villain he swore to fight, leaving Batman to silently carry the burden in the shadows." \
    --instruction_type narrative \
    --config.MAIN_CHARACTER_NAME "Joker, White Knight, Harvey Dent, Two-Face, Batman" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 2 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \

python local_run.py \
    --Video_Path "Dataset/Video/Movie/Paprika.mkv" \
    --Audio_Path "Dataset/Audio/Luv(sic)Pt2.mp3" \
    --Instruction "A vibrant, rhythmic showcase of the seamless transitions between the stoic Dr. Atsuko Chiba and her effervescent alter-ego Paprika, utilizing the track's mellow jazz-hop beat to synchronize the kaleidoscope of colorful transformations, shattering glass, and gravity-defying flights across the dreamscape." \
    --instruction_type object \
    --config.MAIN_CHARACTER_NAME "Paprika, Dr. Atsuko Chiba" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 00:00.0 -e 00:45.0

python local_run.py \
    --Video_Path "Dataset/Video/Movie/Paprika.mkv" \
    --Audio_Path "Dataset/Audio/Luv(sic)Pt2.mp3" \
    --Instruction "A poetic meditation on the healing power of the dream world, using the song's nostalgic lyrics to frame the chaotic parade not as a threat, but as a necessary journey of reconciliation between logic and emotion, culminating in the acceptance of one's true self." \
    --instruction_type narrative \
    --config.MAIN_CHARACTER_NAME "Paprika, Dr. Atsuko Chiba, other main characters" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 00:00.0 -e 00:45.0

python local_run.py \
    --Video_Path "Dataset/Video/Movie/La_La_Land.mkv" \
    --Audio_Path "Dataset/Audio/Norman_fucking_rockwell.mp3" \
    --Instruction "An intimate portrait of Mia's journey from aspiring actress to star, capturing her expressive face through every audition rejection, the hopeful sparkle in her eyes during coffee shop shifts, the vulnerable one-woman show performance, and the confident stride in her final callback. Sync her emotional transformations—tears, laughter, determination—to the song's sweeping piano chords, highlighting close-ups of her colorful dresses swirling as she dances alone, chasing her Hollywood dream." \
    --instruction_type object \
    --config.MAIN_CHARACTER_NAME "Mia, Sebastian" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 01:59.0 -e 02:34.9

python local_run.py \
    --Video_Path "Dataset/Video/Movie/La_La_Land.mkv" \
    --Audio_Path "Dataset/Audio/Norman_fucking_rockwell.mp3" \
    --Instruction "A deconstruction of the Hollywood Ending, using the song's cynical yet tender lyrics to juxtapose the idealized \"Epilogue\" sequence with the gritty reality of the couple's arguments and separation, illustrating that while they were each other's muse, real life—like the song suggests—is far more complicated than a movie script." \
    --instruction_type narrative \
    --config.MAIN_CHARACTER_NAME "Mia, Sebastian" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 01:59.0 -e 02:34.9

python local_run.py \
    --Video_Path "Dataset/Video/Movie/Titanic.mp4" \
    --Audio_Path "Dataset/Audio/CallofSilence.mp3" \
    --Instruction "A visual ode to Rose's transformation from captive aristocrat to liberated soul, capturing her evolution through expressive close-ups—the despair in her eyes at the bow's edge, the awakening joy as she dances freely in third class, the defiant smirk as she spits like a man, and the fierce determination in her face as she survives the freezing waters. Sync her journey of self-discovery to the song's ethereal melody, highlighting her red hair against the ocean, the flowing fabrics of her constraining corsets giving way to simple freedom." \
    --instruction_type object \
    --config.MAIN_CHARACTER_NAME "Rose, Jack" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 01:40.00 -e 02:13.00

python local_run.py \
    --Video_Path "Dataset/Video/Movie/Titanic.mp4" \
    --Audio_Path "Dataset/Audio/CallofSilence.mp3" \
    --Instruction "A heart-wrenching love story chronicling Rose and Jack's fateful encounter and tragic devotion: their first meeting at the ship's bow where he saves her life, the iconic 'I'm flying' moment with arms spread wide against the sunset, their forbidden romance dancing in third class, the tender 'draw me like one of your French girls' scene, the desperate fight for survival as the ship splits apart, and the devastating final moments clinging to a floating door in the freezing Atlantic—Rose's trembling hand holding Jack's as he sacrifices himself, whispering 'you'll die an old woman, warm in your bed,' their love immortalized in the icy waters where they promised to never let go." \
    --instruction_type narrative \
    --config.MAIN_CHARACTER_NAME "Rose, Jack" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 01:40.00 -e 02:13.00

python local_run.py \
    --Video_Path "Dataset/Video/Movie/Interstellar.mkv" \
    --Audio_Path "Dataset/Audio/Moon.mp3" \
    --Instruction "An emotional portrait of Cooper's solitary journey as astronaut and father, capturing the weight of his sacrifice through intimate close-ups—his weathered hands gripping the controls, tears streaming down his face as he watches decades of messages from his children, the determination in his eyes during the desperate docking sequence, and the anguish of aging alone in space. Sync his physical and emotional endurance to the song's atmospheric tones, highlighting his NASA suit reflecting distant stars, the loneliness of his silhouette against infinite black space." \
    --instruction_type object \
    --config.MAIN_CHARACTER_NAME "Cooper, Murph, Astronaut" \
    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 00:00.0 -e 00:36.0

python local_run.py \
    --Video_Path "Dataset/Video/Movie/Interstellar.mkv" \
    --Audio_Path "Dataset/Audio/Moon.mp3" \
    --Instruction "A poignant exploration of love transcending physics, utilizing the track's longing vocals to bridge the heartbreaking parallel edits of Cooper fighting for survival in the tesseract with Murph aging on a dying Earth, illustrating that his desperate drive to leave humanity behind is paradoxically fueled by his desperate need to return to the one person he loves." \
    --instruction_type narrative \
    --config.MAIN_CHARACTER_NAME "Cooper, Murph and Galaxy" \

    --type film \
    --config.AUDIO_MIN_SEGMENT_DURATION 3 \
    --config.AUDIO_MAX_SEGMENT_DURATION 7 \
    --config.AUDIO_TOTAL_SHOTS 50 \
# -s 00:00.0 -e 00:36.0

################################## VLOG Examples ######################################


# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Chongqing.mp4" \
#     --Audio_Path "Dataset/Audio/Let's_go_out_tonight.mp3" \
#     --Instruction "An immersive showcase of Hongya Cave's iconic stilted architecture, capturing the layered yellow lights cascading down the cliffside structure at night, the intricate details of traditional wooden balconies, and the hypnotic reflection of this ancient-modern hybrid in the Jialing River below. Sync the building's rhythmic light patterns and vertical tiers to the upbeat tempo, celebrating this singular architectural marvel as the soul of Chongqing's nightscape." \
#     --instruction_type object \
#     --type vlog \
#     --config.MAIN_CHARACTER_NAME "traditional architecture" \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:43.0 -e 01:27.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Chongqing.mp4" \
#     --Audio_Path "Dataset/Audio/Let's_go_out_tonight.mp3" \
#     --Instruction "Showcase the beautiful scenery of Chongqing's ancient city, specifically capturing the brightly illuminated ancient towers at night and cruise ships slowly gliding along the river, highlighting the perfect fusion of tradition and modernity." \
#     --instruction_type narrative \
#     --config.MAIN_CHARACTER_NAME "beautiful scenery" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:43.0 -e 01:27.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Lisbon.mp4" \
#     --Audio_Path "Dataset/Audio/Mianhuicai.mp3" \
#     --Instruction "A reverent showcase of a grand church's magnificent façade, capturing the intricate stone carvings weathered by centuries, the soaring Gothic arches framed against blue sky, the delicate lacework of limestone columns and ornate motifs, and the interplay of light and shadow across the sacred entrance. Sync the church's majestic architectural details and spiritual presence to the song's rhythmic tempo, celebrating this singular monument as a testament to faith and craftsmanship carved in stone." \
#     --instruction_type object \
#     --config.MAIN_CHARACTER_NAME "church" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:05.0 -e 00:48.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Lisbon.mp4" \
#     --Audio_Path "Dataset/Audio/Mianhuicai.mp3" \
#     --Instruction "A vibrant tapestry of Lisbon's daily life and cultural soul, weaving together bustling street markets filled with fresh fish and colorful produce, locals chatting on tiled doorsteps, vintage trams rattling through narrow cobblestone alleys, laundry fluttering from wrought-iron balconies, and the warm golden hour light bathing pastel-colored buildings. Use the song's rhythm to capture the city's unhurried Mediterranean pace, celebrating Lisbon's authentic charm where tradition, community, and the simple pleasures of life intertwine along the Tagus River." \
#     --instruction_type narrative \
#     --config.MAIN_CHARACTER_NAME "beautiful scenery" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:05.0 -e 00:48.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Egypt.mp4" \
#     --Audio_Path "Dataset/Audio/Mizuiro.mp3" \
#     --Instruction "A majestic tribute to the Great Pyramid standing alone against the vast desert, capturing the monumental limestone blocks weathered by millennia, the sharp geometric apex piercing the blue sky, and the interplay of harsh sunlight creating dramatic shadows across its ancient faces. Sync the pyramid's timeless solidity and mathematical precision to the soothing melodic tones, celebrating this singular wonder's eternal presence in the endless sands." \
#     --instruction_type object \
#     --config.MAIN_CHARACTER_NAME "Pyramid" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:26.0 -e 01:16.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Egypt.mp4" \
#     --Audio_Path "Dataset/Audio/Mizuiro.mp3" \
#     --Instruction "The exhibit showcases the immense size of the pyramids and the insignificance of humankind, while also highlighting the desolate surrounding environment, thus embodying the pyramids' wonder and awe." \
#     --instruction_type narrative \
#     --config.MAIN_CHARACTER_NAME "beautiful scenery" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:26.0 -e 01:16.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Kyoto.mp4" \
#     --Audio_Path "Dataset/Audio/rumination.mp3" \
#     --Instruction "A meditative focus on Kiyomizu-dera (Pure Water Temple) blanketed in pristine snow, capturing the iconic wooden stage jutting out from the hillside, the delicate accumulation of snow on vermilion pillars and traditional rooftops, and the panoramic view of snow-covered Kyoto spreading below. Sync the temple's serene stillness and ancient wooden architecture to the calm, reflective mood of the song, celebrating this singular monument's tranquil beauty in winter's embrace." \
#     --instruction_type object \
#     --config.MAIN_CHARACTER_NAME "Temple" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:00.0 -e 00:43.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Kyoto.mp4" \
#     --Audio_Path "Dataset/Audio/rumination.mp3" \
#     --Instruction "In snow-covered Kyoto, the Japanese gardens take on a more solemn air under the blanket of white, while the temples gain an added touch of Zen tranquility" \
#     --instruction_type narrative \
#     --config.MAIN_CHARACTER_NAME "beautiful scenery" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 5 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 10 \
#     --config.AUDIO_TOTAL_SHOTS 30 \
# # -s 00:00.0 -e 00:43.0

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Switzerland.mp4" \
#     --Audio_Path "Dataset/Audio/golden_hour.mp3" \
#     --Instruction "An intimate portrait of a traditional Swiss alpine chalet, capturing the weathered timber beams and intricate wooden carvings, the flower boxes overflowing with vibrant geraniums beneath shuttered windows, the stone foundation anchoring the structure to the hillside, and the rustic slate roof catching golden hour light. Sync the building's timeless craftsmanship and cozy details—hand-carved balconies, decorative woodwork, and warm wooden textures—to the song's glowing tones, celebrating this singular piece of mountain architecture as a testament to Swiss pastoral life." \
#     --instruction_type object \
#     --config.MAIN_CHARACTER_NAME "alpine chalet" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 3 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 7 \
#     --config.AUDIO_TOTAL_SHOTS 50 \
#     --config.AUDIO_WEIGHT_PITCH 2 \
# # -s 00:39.00 -e 01:16.00

# python local_run.py \
#     --Video_Path "Dataset/Video/VLOG/Switzerland.mp4" \
#     --Audio_Path "Dataset/Audio/golden_hour.mp3" \
#     --Instruction "A breathtaking celebration of the Swiss Alps' natural grandeur, weaving together sweeping vistas of snow-capped peaks piercing azure skies, pristine mountain lakes reflecting towering summits, rolling meadows carpeted with wildflowers, cascading waterfalls tumbling down rocky cliffs, and gentle streams meandering through emerald valleys. Use the song's warm, glowing melody to capture nature's symphony—from the first light kissing mountain ridges to mist drifting through pine forests—illustrating the raw, untouched beauty of the alpine wilderness in its purest form." \
#     --instruction_type narrative \
#     --config.MAIN_CHARACTER_NAME "beautiful scenery" \
#     --type vlog \
#     --config.AUDIO_MIN_SEGMENT_DURATION 3 \
#     --config.AUDIO_MAX_SEGMENT_DURATION 7 \
#     --config.AUDIO_TOTAL_SHOTS 50 \
#     --config.AUDIO_WEIGHT_PITCH 2 \
# # -s 00:39.00 -e 01:16.00


  



