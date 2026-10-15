import os
import pathlib
import time
import datetime

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import json
import random
import csv
import concurrent.futures
from multiprocessing import freeze_support

from keras import layers
from keras import models
from keras import mixed_precision

seed = 6969
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

pre_segment_size = 12
post_segment_size = 12
prediction_size = 8
segment_size = pre_segment_size + post_segment_size + prediction_size

note_size = 49
replays_simple_dir = pathlib.Path("decoded-replays")
replays_dir = replays_simple_dir
leaderboards_dir = pathlib.Path("leaderboards")
maps_dir = pathlib.Path("maps")
speed_stuff = False

def get_leaderboard_replays(percentage=100, split=20):

    leaderboard_ids = np.array(tf.io.gfile.listdir(str(replays_dir)))
    random.shuffle(leaderboard_ids)
    leaderboard_ids = leaderboard_ids[:int(len(leaderboard_ids)*percentage/100)]
    val_leaderboard_ids = leaderboard_ids[:int(len(leaderboard_ids)*split/100)]
    val_leaderboard_ids
    # val_leaderboard_ids = []

    train_data = []
    val_data = []
    for leaderboard_id in leaderboard_ids:
        if leaderboard_id == "219625":
            continue
        
        if leaderboard_id in val_leaderboard_ids:
            val_data.append(leaderboard_id)
        else:
            train_data.append(leaderboard_id)

    return train_data, val_data


def read_json_file(file):
    try:
        with open(file, "r", encoding="utf8", errors="ignore") as f:
            file_content = f.read()
            if len(file_content) < 100:
                return None
            json_content = json.loads(file_content)
            return json_content
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e)
        print(file)
        raise


def get_replay_notes(replay, njs, time_scale):
    notes = []

    prev_zero_note_time = 0
    prev_one_note_time = 0
    # for note_info, score, note_time in sorted(replay, key=lambda item: item[2]):

    for note_time, note_info, prediction in replay:
        type = note_info[-1]
        score, speed = prediction

        # TODO: use map data for note positions and timings to not have to exclude misses (misses are registered much later, which messes up the timings)
        if score < 0:
            continue

        # NOTE: 0-100 score range is rare and often happens for tracking problems that are not important here
        # would be good to replace this with acc component only and potentially learn all both acc and swing angles
        # but need different format replay files for that
        # score = max(0, score - 100)

        delta_to_zero = note_time - prev_zero_note_time
        delta_to_one = note_time - prev_one_note_time

        if delta_to_zero < 0 or delta_to_one < 0:
            print(f"{delta_to_zero} {delta_to_one}")

        if type == "0":
            prev_zero_note_time = note_time
            note = preprocess_note(prediction, delta_to_zero,
                                   delta_to_one, note_info, njs, time_scale)
            notes.append(note)
        if type == "1":
            prev_one_note_time = note_time
            note = preprocess_note(prediction, delta_to_one,
                                   delta_to_zero, note_info, njs, time_scale)
            notes.append(note)

    return notes


def preprocess_note(prediction, delta, delta_other, note_info, njs, time_scale):
    # NOTE: timing increases difficulty not linearly and caps out at ~2 seconds
    # no idea if such parameters can be learned by neural networks without adding scaling like I did right here
    # delta = int(delta*1000)/1000
    # delta_other = int(delta_other*1000)/1000

    # NOTE: timing increases difficulty not linearly and caps out at ~2 seconds
    # no idea if such parameters can be learned by neural networks without adding scaling like I did right here
    delta = delta/time_scale
    delta_other = delta_other/time_scale
    njs = njs*time_scale
    
    delta_long = max(0, 2 - delta)/2
    delta_other_long = max(0, 2 - delta_other)/2
    delta_short = max(0, 0.5 - delta)*2
    delta_other_short = max(0, 0.5 - delta_other)*2

    col_number = int(note_info[0])
    row_number = int(note_info[1])
    direction_number = int(note_info[2])
    color = int(note_info[3])

    row_col = [0] * 4 * 3
    direction = [0] * 10
    
    row_col2 = [0] * 4 * 3
    direction2 = [0] * 10
    
    row_col[col_number * 3 + row_number] = 1
    direction[direction_number] = 1

    # color_arr = [0] * 2
    # color_arr[color] = 1

    response = []

    if color == 0:
        response.extend(row_col)
        response.extend(direction)
        response.extend(row_col2)
        response.extend(direction2)
        response.extend([
            delta_short,
            delta_long,
        ])
        response.extend([
            delta_other_short,
            delta_other_long,
        ])
    if color == 1:
        response.extend(row_col2)
        response.extend(direction2)
        response.extend(row_col)
        response.extend(direction)
        response.extend([
            delta_other_short,
            delta_other_long,
        ])
        response.extend([
            delta_short,
            delta_long,
        ])
        
    # response.extend(row_col)
    # response.extend(direction)
    # response.extend(color_arr)

    response.extend([
        njs/30,
        prediction
    ])

    return response
# print(len(preprocess_note(0.5, 0.5, 0.5, "0000", 0.5, 1)))

def create_segments(notes):
    empty_res = ([], [])
    if len(notes) < prediction_size:
        return empty_res

    segments = []
    predictions = []
    for i in range(len(notes)-prediction_size+1):
        if i % prediction_size != 0:
            continue
                
        pre_slice = notes[max(0, i-pre_segment_size):i]
        slice = notes[i:i+prediction_size]
        post_slice = notes[i+prediction_size:i+prediction_size+post_segment_size]

        # NOTE: using relative score can be good to find relative difficulty of the notes more fairly
        # because good players will always get higher acc and worse players will do badly even on easy patterns

        pre_segment = [np.array(note[:-1]) for note in pre_slice]
        if len(pre_segment) < pre_segment_size:
            pre_segment[0:0] = [np.zeros(note_size, dtype=np.float32) for i in range(pre_segment_size - len(pre_segment))]
            
        segment = [np.array(note[:-1]) for note in slice]

        post_segment = [np.array(note[:-1]) for note in post_slice]
        if len(post_segment) < post_segment_size:
            post_segment.extend([np.zeros(note_size, dtype=np.float32) for i in range(post_segment_size - len(post_segment))])


        # fix this pls
        prediction = [note[-1][1] if speed_stuff else note[-1][0] for note in slice]

        final_segment = []
        final_segment.extend(pre_segment)
        final_segment.extend(segment)
        final_segment.extend(post_segment)
        segments.append(final_segment)
        
        predictions.append(prediction)

    return segments, predictions


def get_replays_for_leaderboard(leaderboard_id):
    for replay_file in glob.glob(f'{replays_dir}/{leaderboard_id}/*.dat.json'):
        replay = read_json_file(replay_file)
        if replay is None or replay["info"]["st"] != 0 or replay["info"]["leftHanded"]:
            continue
        replay["fileName"] = replay_file
        yield replay
        
        
def get_leaderboard_data(leaderboard_id, minPlays):
    leaderboard_info_file = f'{replays_simple_dir}/{leaderboard_id}/leaderboard-info.json'

    leaderboard_info = read_json_file(leaderboard_info_file)
    beatsaver_key = leaderboard_info["info"]["beatsaverKey"]
    difficulty = leaderboard_info["info"]["difficulty"]["difficulty"]
    number_of_plays = leaderboard_info["info"]["plays"]
    
    if number_of_plays < minPlays:
        raise Exception()
    
    map_path = f'{maps_dir}/{beatsaver_key}'
    if not os.path.isdir(map_path):
        # Download map zip from BeatSaver
        import zipfile
        import io
        import requests

        url = f"https://beatsaver.com/api/maps/id/{beatsaver_key}"
        data = json.loads(requests.get(url).text)
        map_link = data["versions"][0]["downloadURL"]

        r = requests.get(map_link)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(map_path)
    
    map_info_file = glob.glob(f'{map_path}/*nfo.dat')[0]

    with open(map_info_file, "r", encoding="utf8", errors="ignore") as f:
        file_content = f.read()
        map_info = json.loads(file_content)
        bpm = map_info["_beatsPerMinute"]
        time_scale = 60/bpm

        for beatmap_set in map_info["_difficultyBeatmapSets"]:
            if beatmap_set["_beatmapCharacteristicName"] != "Standard":
                continue

            for beatmap in beatmap_set["_difficultyBeatmaps"]:
                if beatmap["_difficultyRank"] == difficulty:
                    njs = float(beatmap["_noteJumpMovementSpeed"])
                    map_file_name = beatmap["_beatmapFilename"]
                    with open(map_info_file.replace("Info.dat", map_file_name).replace("info.dat", map_file_name), "r", encoding="utf8", errors="ignore") as map_file:
                        map_file_content = map_file.read()
                        map_file_json = json.loads(map_file_content)
                        map_notes = sorted(list(map(lambda n: (n["_time"]*time_scale, f"{n['_lineIndex']}{n['_lineLayer']}{n['_cutDirection']}{n['_type']}"), filter(lambda n: n['_type'] == 1 or n['_type'] == 0, map_file_json["_notes"]))), key=lambda x: (x[0], x[1]))

    replays = get_replays_for_leaderboard(leaderboard_id)

    data = [njs, number_of_plays]

    return data, map_notes, replays


def get_replay_speeds(angleFrames, noteTimes, movingAverageTime):
    angleFramesIter = 0
    
    replay_note_saber_speeds = [[] for _ in noteTimes]
    
    for index, noteTime in sorted(enumerate(noteTimes), key=lambda x: x[1]):
        while noteTime > angleFrames[angleFramesIter][0]:
            angleFramesIter += 1
        
        lastAngleRight = (angleFrames[angleFramesIter][1])/(angleFrames[angleFramesIter][0] - angleFrames[angleFramesIter - 1][0])*(noteTime-angleFrames[angleFramesIter - 1][0])
        lastAngleLeft = (angleFrames[angleFramesIter][2])/(angleFrames[angleFramesIter][0] - angleFrames[angleFramesIter - 1][0])*(noteTime-angleFrames[angleFramesIter - 1][0])

        reverseAngleFramesIter = 1
        currNoteTime = noteTime - movingAverageTime
        while True:
            currIter = angleFramesIter-reverseAngleFramesIter
            if angleFrames[currIter][0] < currNoteTime:
                firstAngleRight = (angleFrames[currIter+1][1])/(angleFrames[currIter+1][0] - angleFrames[currIter][0])*(angleFrames[currIter+1][0] - (currNoteTime))
                firstAngleLeft = (angleFrames[currIter+1][2])/(angleFrames[currIter+1][0] - angleFrames[currIter][0])*(angleFrames[currIter+1][0] - (currNoteTime))
                break
            else:
                reverseAngleFramesIter += 1
        
        totalAngleRight = firstAngleRight + lastAngleRight
        totalAngleLeft = firstAngleLeft + lastAngleLeft
        for frameTime, angleRight, angleLeft in angleFrames[currIter+2:angleFramesIter]:
            totalAngleRight += angleRight
            totalAngleLeft += angleLeft

        angleSpeedRight = totalAngleRight/movingAverageTime
        angleSpeedLeft = totalAngleLeft/movingAverageTime
        
        replay_note_saber_speeds[index] = [angleSpeedRight, angleSpeedLeft]
    
    rights = [v[0] for v in replay_note_saber_speeds]
    lefts = [v[1] for v in replay_note_saber_speeds]
    print(sum(rights)/len(rights), sum(lefts)/len(lefts))
    
    return replay_note_saber_speeds


def get_replay_accelerations(angleFrames, noteTimes, movingAverageTime):
    angle_speeds = []
    for i, [time, right, left] in enumerate(angleFrames):
        if i < 1:
            continue
        prev_time, prev_right, prev_left = angleFrames[i-1]
        if time-prev_time == 0:
            continue
        t_d = time - prev_time
        right_d = abs(right - prev_right)
        left_d = abs(left - prev_left)
        angle_speeds.append([time, right_d/t_d, left_d/t_d])
    
    angle_accelerations = []
    for i, [time, right, left] in enumerate(angle_speeds):
        if i < 1:
            continue
        prev_time, prev_right, prev_left = angle_speeds[i-1]
        if time-prev_time == 0:
            continue
        t_d = time - prev_time
        right_d = abs(right - prev_right)
        left_d = abs(left - prev_left)
        angle_accelerations.append([time, right_d/t_d, left_d/t_d])

    angleFrames = angle_accelerations
    angleFramesIter = 0
    
    replay_note_saber_speeds = [[] for _ in noteTimes]
    
    for index, noteTime in sorted(enumerate(noteTimes), key=lambda x: x[1]):
        while noteTime > angleFrames[angleFramesIter][0]:
            angleFramesIter += 1
        
        lastAngleRight = (angleFrames[angleFramesIter][1])/(angleFrames[angleFramesIter][0] - angleFrames[angleFramesIter - 1][0])*(noteTime-angleFrames[angleFramesIter - 1][0])
        lastAngleLeft = (angleFrames[angleFramesIter][2])/(angleFrames[angleFramesIter][0] - angleFrames[angleFramesIter - 1][0])*(noteTime-angleFrames[angleFramesIter - 1][0])

        reverseAngleFramesIter = 1
        currNoteTime = noteTime - movingAverageTime
        while True:
            currIter = angleFramesIter-reverseAngleFramesIter
            if angleFrames[currIter][0] < currNoteTime:
                firstAngleRight = (angleFrames[currIter+1][1])/(angleFrames[currIter+1][0] - angleFrames[currIter][0])*(angleFrames[currIter+1][0] - (currNoteTime))
                firstAngleLeft = (angleFrames[currIter+1][2])/(angleFrames[currIter+1][0] - angleFrames[currIter][0])*(angleFrames[currIter+1][0] - (currNoteTime))
                break
            else:
                reverseAngleFramesIter += 1
        
        totalAngleRight = firstAngleRight + lastAngleRight
        totalAngleLeft = firstAngleLeft + lastAngleLeft
        items = 2
        for frameTime, angleRight, angleLeft in angleFrames[currIter+2:angleFramesIter]:
            items += 1
            totalAngleRight += angleRight
            totalAngleLeft += angleLeft

        angleSpeedRight = totalAngleRight/items
        angleSpeedLeft = totalAngleLeft/items
        
        replay_note_saber_speeds[index] = [angleSpeedRight, angleSpeedLeft]
           
    rights = [v[0] for v in replay_note_saber_speeds]
    lefts = [v[1] for v in replay_note_saber_speeds]
    print(sum(rights)/len(rights), sum(lefts)/len(lefts))
    print()
        
    return replay_note_saber_speeds


def get_replay_total_score(replay):
    return sum([max(1-1/0.3*s, 0) for s, s1, s2 in replay["scores"]])


def get_sorted_filtered_replays(replays):
    highest_total_score = -1
    for total_score, replay in sorted([(get_replay_total_score(replay), replay) for replay in replays], key=lambda x: x[0], reverse=True):
        if highest_total_score == -1:
            highest_total_score = total_score
        if total_score/highest_total_score > 0.94:
            yield replay


def preprocess_leaderboard_replays(leaderboard_id, skip_replays=False, time_scale=1):
    count = 0
    skip = False
    min_map_plays = 400
    # empty_res = []
    empty_res = ([], [])
    
    try:
        map_data, map_notes, replays = get_leaderboard_data(leaderboard_id, min_map_plays)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        return empty_res

    if (map_data[0] > 100) and not skip_replays:
        return empty_res

    # note time, note info, scores
    the_thing = [(note_time, note_info, []) for note_time, note_info in map_notes]

    if not skip_replays:
        for replay in get_sorted_filtered_replays(replays):
            note_infos = replay["noteInfos"]
            
            scores = []

            if speed_stuff:
                print(replay["fileName"])
                # speeds = get_replay_speeds(replay["angleFrames"], replay["noteTime"], 0.1)
                speeds = get_replay_accelerations(replay["angleFrames"], replay["noteTime"], 0.1)
                for score, speed, note_info in zip(replay["scores"], speeds, note_infos):
                    scores.append([score[0], speed[0] if int(note_info[-1]) == 1 else speed[1]])
            else:
                for score in replay["scores"]:
                    scores.append([score[0], 0])
            
            left_handed = replay["info"]["leftHanded"]
            if left_handed:
                continue
                # no worky
                note_infos_mirrored = []
                for note_info in note_infos:
                    if(len(note_info) > 4):
                        note_infos_mirrored.append(note_info)
                        continue
                    col = int(note_info[0])
                    row = int(note_info[1])
                    dir = int(note_info[2])
                    color = int(note_info[3])
                    new_note_info = f"{3-col}{row}{dir if dir < 2 else (dir + 1 if dir%2 == 0 else -1)}{1-color}"
                    note_infos_mirrored.append(new_note_info)
                note_infos = note_infos_mirrored

            if(count > 15):
                break

            indexes = {}
            num_elements = 0
            for note_info, score in zip(note_infos, scores):
                if len(note_info) > 4 or score[0] < -3:
                    continue

                num_elements += 1
                if note_info in indexes:
                    indexes[note_info].append(score)
                else:
                    indexes[note_info] = [score]

            if num_elements < len(map_notes):
                continue
            try:
                for note_time, note_info, scores in the_thing:
                    score = indexes[note_info].pop(0)

                    if score[0] > 0:
                        scores.append([score[0], score[1]/100])
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                skip = True
                break
            
            count += 1
    
    if skip:
        return empty_res

    if count < 9 and not skip_replays:
        return empty_res

    asd = []
    for note_time, note_info, scores in the_thing:
        length = len(scores)
        if length <= 0:
            continue
        
        # scores = sorted(scores, reverse=True)[int(len(scores)*0.5):]
        score = sum(sorted([max(1-1/0.3*s[0], 0) for s in scores])[1:-1])/(length-2)
        speeds = sorted([s[1]*0.1 for s in scores])[3:-3]
        speed = sum(speeds)/len(speeds)*time_scale

        asd.append((note_time, note_info, [score, speed]))

    notes = get_replay_notes(asd, map_data[0], time_scale)
    return create_segments(notes)

def get_map_data(beatsaver_key, difficulty):
    map_info_file = glob.glob(f'{maps_dir}/{beatsaver_key}/*nfo.dat')[0]
    njs = None
    map_notes = None
    
    with open(map_info_file, "r", encoding="utf8", errors="ignore") as f:
        file_content = f.read()
        map_info = json.loads(file_content)
        bpm = map_info["_beatsPerMinute"]
        time_scale = 60/bpm

        for beatmap_set in map_info["_difficultyBeatmapSets"]:
            if beatmap_set["_beatmapCharacteristicName"] != "Standard":
                continue

            for beatmap in beatmap_set["_difficultyBeatmaps"]:
                if beatmap["_difficultyRank"] == difficulty:
                    njs = float(beatmap["_noteJumpMovementSpeed"])
                    map_file_name = beatmap["_beatmapFilename"]
                    with open(map_info_file.replace("Info.dat", map_file_name).replace("Info.dat", map_file_name), "r", encoding="utf8", errors="ignore") as map_file:
                        map_file_content = map_file.read()
                        map_file_json = json.loads(map_file_content)
                        map_notes = sorted(list(map(lambda n: (n["_time"]*time_scale, f"{n['_lineIndex']}{n['_lineLayer']}{n['_cutDirection']}{n['_type']}"), filter(lambda n: n['_type'] == 1 or n['_type'] == 0, map_file_json["_notes"]))), key=lambda x: (x[0], x[1]))

    return njs, map_notes


def preprocess_map(beatsaver_key, difficulty, scale):
    empty_response = ([], [])
    njs, map_notes = get_map_data(beatsaver_key, difficulty)
    if njs == None or map_notes == None:
        return empty_response
    
    # note time, note info, saber speeds, scores
    asd = [(note_time, note_info, [0, 0]) for note_time, note_info in map_notes]
    
    notes = get_replay_notes(asd, njs, scale)
    return create_segments(notes)

def generate_data(leaderboard_ids, num_threads, time_scales, disable_tqdm=False):
    segments = []
    predictions = []

    cancel = False

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Dictionary to store future to leaderboard_id mapping
        future_to_leaderboard_id = {}
        for time_scale in time_scales:
            for leaderboard_id in leaderboard_ids:
                future = executor.submit(preprocess_leaderboard_replays, leaderboard_id, False, time_scale)
                future_to_leaderboard_id[future] = leaderboard_id

        # Wrapping futures in tqdm for progress tracking
        for future in tqdm(concurrent.futures.as_completed(future_to_leaderboard_id), total=len(future_to_leaderboard_id), disable=disable_tqdm):
            leaderboard_id = future_to_leaderboard_id[future]
            try:
                if cancel:
                    future.cancel()
                    continue

                segment, prediction = future.result()

                segments.extend(segment)
                predictions.extend(prediction)
            except (KeyboardInterrupt, SystemExit):
                cancel = True
            except Exception as e:
                print(f"{leaderboard_id} - {e}")
                continue

    return np.array(segments), np.array(predictions)

def generate_data_sync(leaderboard_ids, time_scales, disable_tqdm=False):
    segments = []
    predictions = []
    
    for leaderboard_id in tqdm(leaderboard_ids, disable=disable_tqdm):
        try:
            # Call the preprocess function directly for each leaderboard ID and time scale
            segment, prediction = preprocess_leaderboard_replays(leaderboard_id, False)
            
            segments.extend(segment)
            predictions.extend(prediction)
        except (KeyboardInterrupt, SystemExit):
            # Handling for keyboard interrupt to stop processing further
            break
        except Exception as e:
            # Exception handling for any errors during the preprocess call
            print(f"{leaderboard_id} - {e}")
            continue

    return np.array(segments), np.array(predictions)