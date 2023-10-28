import os
from tqdm import tqdm
from torch import cuda
import os
import json

from Utils.TextProcess import CombineSegmentsToWordList, ExtractcKeyphrasesFromText, CombineKeyphrases, \
                            SplitToSegments, SplitToShorts
from Utils.Download import DownloadVideoAndSub
from Utils.SpeechToText import ExtractTimestampsWhisper, CleanTimestamps
from pytube import Playlist
from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip, TextClip
from Utils.VideoEditing import CenteredWithBlurred
from Utils.Helper import CreateFolder

VIDEO_SIZE = (1080,1920)
DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
BASE_PATH = "videos"
CreateFolder(BASE_PATH)

def BuildSegmentText(words:list):
    to_ret = ""
    for word in words:
        text = word['text']
        if word['key']:
            to_ret+=f'<span foreground="yellow">{text} </span>'
        else:
            to_ret+=f'<span foreground="white">{text} </span>'
    return to_ret

def HandleVideoURL(url:str):
    # get video and audio
    print(' Downloading video')
    vid_id = DownloadVideoAndSub(url,BASE_PATH)

    vid_folder_path = os.path.join(BASE_PATH,vid_id)
    video_path = os.path.join(vid_folder_path,vid_id+'.mp4')
    audio_path = os.path.join(vid_folder_path,vid_id+'.mp3')
    shorts_folder_path = os.path.join(vid_folder_path,'shorts')
    CreateFolder(shorts_folder_path)
    shorts_json_path = os.path.join(vid_folder_path,'shorts.json')
    whisper_json_path = os.path.join(vid_folder_path,'whisper.json')
    print("   Downloaded data")

    if not os.path.exists(shorts_json_path):
        # getting text
        segments = []
        if not os.path.exists(whisper_json_path):
            segments, all_text_whisper = ExtractTimestampsWhisper(audio_path)
            segments = CleanTimestamps(segments)
            with open(whisper_json_path,'w+') as f:
                f.write(json.dumps(segments, indent=4))
        else:
            with open(whisper_json_path) as f:
                segments = json.load(f)
        print("   Exctracted Timestamps")

        # Text Process
        ## geting keywords
        word_time_stamps = CombineSegmentsToWordList(segments)
        keyphrases_in_text = ExtractcKeyphrasesFromText(all_text_whisper)
        word_time_stamps = CombineKeyphrases(keyphrases_in_text, word_time_stamps)
        print("   Extracted Keyphrases")

        ## spliting to segments
        segments = SplitToSegments(word_time_stamps)
        print("   Split to segments")

        ## splitting to shorts
        shorts = SplitToShorts(segments)
        print("   Finished text process")

        ## saving to json
        with open(shorts_json_path,'w+') as f:
            f.write(json.dumps(shorts, indent=4))
    return {
        'vid_folder_path': vid_folder_path,
        'video_path': video_path,
        'audio_path': audio_path,
        'shorts_folder_path': shorts_folder_path,
        'shorts_json_path': shorts_json_path,
    }

def DownloadAndProcessFromURL(url:str):
    to_ret = []
    if "watch?v=" in url:
        print('Video')
        to_ret.append(HandleVideoURL(url))
    elif "playlist?list=" in url:
        print('Playlist')
        playlist_url = url
        playlist = Playlist(playlist_url)

        # iterate through all videos in the playlist and print their URLs
        n_videos = len(playlist.videos)
        for idx, video in enumerate(playlist.videos):
            print(f"{idx+1}/{n_videos}")
            try:
                to_ret.append(HandleVideoURL(video.watch_url))
            except Exception as e:
                print(f"error with {video.watch_url}")
    else:
        raise Exception("URL is not a video nor a playlist")
    
    with open(os.path.join(BASE_PATH,'videos_data.json'), 'w+') as f:
        f.write(json.dumps(to_ret, indent=4))
    return to_ret

def GenerateShortsFromVideoData(data:dict, shorts:list):
    to_ret = []
    # Load the video
    video = VideoFileClip(data['video_path'])
    fps = video.fps
    audio = AudioFileClip(data['audio_path'])

    n_shorts = len(shorts)
    for short_id, short in enumerate(shorts):
        start_time, end_time = short['start'], short['end']
        final_short_path = os.path.join(data['shorts_folder_path'], str(short_id) + '.mp4')

        if not os.path.exists(final_short_path):
            print(f'    {short_id+1}/{n_shorts}')

            short_audio = audio.subclip(start_time, end_time)
            short_video = CenteredWithBlurred(video.subclip(start_time, end_time),VIDEO_SIZE)
            print('    Blurred And Centered')

            # Define the text clip
            text_clips = []
            for segment in tqdm(short['segments']):
                text = BuildSegmentText(segment['words']).replace('&', 'and')
                try:
                    text_clip = (TextClip(text, fontsize=60, font='Impact',
                                        size=VIDEO_SIZE, method='pango')
                        .set_duration(segment['end'] - segment['start'])
                        .set_start(segment['start']-start_time))
                except Exception as e:
                    print(e)
                    print(text)

                text_width = text_clip.w
                x_pos = (VIDEO_SIZE[0] - text_width) // 2
                y_pos = int(VIDEO_SIZE[1]*0.7)
                text_clip = text_clip.set_position((x_pos, y_pos))
                text_clips.append(text_clip)
                
            full_text_clip = CompositeVideoClip(text_clips)
            print('    Generated Text Clips')
            final_clip = CompositeVideoClip([short_video, full_text_clip])
            print('    Combined Video And Text Clip')
            final_clip = final_clip.set_audio(short_audio)
            print('    Combined Video And Audio')
            final_clip.write_videofile(final_short_path, codec='libx264', fps=fps, threads=32)
        to_ret.append({'path':final_short_path})
    return to_ret

def VideosDataToShorts(videos_data:list, shorts_paths_file:str):
    shorts_paths = []
    n_videos = len(videos_data)
    for idx, video_data in enumerate(videos_data):
        shorts = []
        with open(video_data['shorts_json_path']) as f:
            shorts = json.load(f)
        shorts.sort(key=lambda x: x['start']) 
        print(f"{idx+1}/{n_videos}")
        shorts_paths += GenerateShortsFromVideoData(video_data, shorts)

    with open(shorts_paths_file, 'w+') as f:
        f.write(json.dumps(shorts_paths, indent=4))
    return shorts_paths

def HandleURL(url:str, shorts_paths_file:str):
    # downloading videos and extracting data
    print("getting videos data...")
    videos_data = DownloadAndProcessFromURL(url)

    # generating shorts
    print("spliting data to shorts...")
    shorts_paths = VideosDataToShorts(videos_data, shorts_paths_file)

    return shorts_paths