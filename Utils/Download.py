from pytube import YouTube, Playlist
from moviepy.editor import VideoFileClip
from Utils.Helper import CreateFolder
from math import ceil
import os
from bing_image_downloader import downloader

MAX_REL_RES = 1080

def GetResolutionAudioDict(yt:YouTube):
  to_ret = []
  for stream in yt.streams:
    if stream.includes_video_track:
      to_ret.append({
          "res":int(stream.resolution[:-1]),
          "audio": stream.is_progressive,
          "stream":stream
      })
  to_ret = sorted(to_ret, key=lambda x: x['res'])[::-1]
  return to_ret

def GetHighestResWithAudio(arr:list):
  max = 0
  stream = None
  for dic in arr:
    if dic["res"]>max and dic["audio"]:
      max = dic["res"]
      stream = dic["stream"]
    if max>=MAX_REL_RES:
      break
  return stream

def ExtractAudioFromVideo(video_path, output_path):
  video = VideoFileClip(video_path)
  audio = video.audio
  
  audio.write_audiofile(output_path)

def DownloadVideo(yt:YouTube,save_path,video_id,highest_res=True):
  downloaded_flag = os.path.join(save_path,'downloaded.txt')
  if os.path.exists(downloaded_flag):
    return
  res_audio = GetResolutionAudioDict(yt)
  video = GetHighestResWithAudio(res_audio)
  
  if video is None or highest_res:
    video = yt.streams.filter(file_extension='mp4',type='video').order_by("resolution")[-1]

  video_filename = f"{video_id}.mp4"
  video_filepath = os.path.join(save_path,video_filename)

  # download video
  video.download(output_path=save_path, filename=video_filename)

  audio_filename = f"{video_id}.mp3"
  audio_filepath = os.path.join(save_path,audio_filename)
  if not video.includes_audio_track:
    audio = yt.streams.filter(type='audio').get_audio_only()
    if audio is not None:
      audio.download(output_path=save_path, filename=audio_filename)
  else:
    # extract audio from file
    ExtractAudioFromVideo(video_filepath,audio_filepath)
  with open(downloaded_flag,'w+') as f:
    f.write('a')

def DownloadVideoAndSub(url, save_path):
  yt = YouTube(url)
  video_id = yt.video_id
  save_path = os.path.join(save_path,video_id)
  CreateFolder(save_path)
  
  DownloadVideo(yt,save_path,video_id)

  return video_id

def GetTotalURLSDuration(urls:list):
    try:
      to_ret = 0
      for url in urls:
        youtube = YouTube(url)
        to_ret += youtube.length
      return ceil(to_ret/60)
    except:
       return None

def GetAllURLS(url:str):
    if "watch?v=" in url:
      return [url]
    elif "playlist?list=" in url:
        playlist = Playlist(url)
        return playlist.video_urls
    
def DownloadImagesFromSegments(segments: list, images_path:str):
  downloaded_keyphrases = os.listdir(images_path)
  for segment in segments:
    for word in segment['words']:
      if word['key'] and word['text'].lower() not in downloaded_keyphrases:
        query_string = word['text'].lower()
        try:
          downloader.download(query_string, limit=5, output_dir=images_path, adult_filter_off=True, force_replace=False, timeout=60, verbose=False)
        except:
          continue
        downloaded_keyphrases.append(word['text'].lower())