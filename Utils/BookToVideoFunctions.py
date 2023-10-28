from tqdm import tqdm
import numpy as np
import cv2
from os import remove
from os.path import join
import textwrap
from moviepy.editor import AudioFileClip
from Utils.Helper import SelectImageByKeyphrase
from Utils.Download import DownloadImagesFromSegments
from Utils.TextProcess import CombineSegmentsToWordList, ExtractcKeyphrasesFromText, CombineKeyphrases, \
                            SplitToSegments
from Utils.SpeechToText import ExtractTimestampsWhisper, CleanTimestamps
from Utils.TextProcess import AddBlanksAndStrech
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
from google.cloud import texttospeech
from Utils.Helper import CreateFolder

IMAGES_PATH = 'images'
CreateFolder(IMAGES_PATH)

VIDEO_SIZE = (1920,1080) # h,w
NP_SHAPE = (1920,1080,3) # h,w,c
FPS = 60
TITLE_SECTION_DURATION = 1.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_SCALE = 3
TEXT_SCALE = 2
TEXT_COLOR = (255,255,255)  # RGB values, here white color.
TEXT_THICNESS = 4  # Thickness of the font lines
TITLE_THICNESS = 8  # Thickness of the font lines
BORDER_THICKNESS = 15  # Thickness of the border
TEXT_GAP = 60
TITLE_GAP = 120
TEXT_WIDTH = 20
TITLE_WIDTH = 14
SOUND_HZ = 44100

def GenerateAudioFromText(text, output_file):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-I" # "en-US-Studio-M"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.15  # Increase the speaking rate for faster speech
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open(output_file, "wb") as out:
        out.write(response.audio_content)

def WriteVideoPart(frame, duration, out):
    # # set image duration
    n_frames_to_write = round(duration*FPS)
    for _ in range(n_frames_to_write):
        out.write(frame.astype(np.uint8))

def AddTextToImage(bg_img_final,text, section_title=False):
    # Break the text into lines
    scale = TEXT_SCALE
    y_buffer = bg_img_final.shape[0]*0.3
    gap = TEXT_GAP
    width = TEXT_WIDTH
    thickness = TEXT_THICNESS
    if section_title:
        y_buffer = 0
        scale = TITLE_SCALE
        gap = TITLE_GAP
        thickness = TITLE_THICNESS
        width = TITLE_WIDTH
    
    wrapper = textwrap.TextWrapper(width=width) # Adjust as necessary for your desired width
    word_list = wrapper.wrap(text=text)

    for i, line in enumerate(word_list):
        textsize = cv2.getTextSize(line, FONT, scale, BORDER_THICKNESS)[0]
        
        y = int((bg_img_final.shape[0] + textsize[1]) / 2 + y_buffer) + i * gap
        x = int((bg_img_final.shape[1] - textsize[0]) / 2)

        # add text
        if not section_title:
            cv2.putText(bg_img_final, line, (x, y), FONT, scale, (0,0,0), BORDER_THICKNESS, lineType=cv2.LINE_AA)
        cv2.putText(bg_img_final, line, (x, y), FONT, scale, TEXT_COLOR, thickness, lineType=cv2.LINE_AA)

def BuildSectionTitle(section_title, out):
    bg_img_final = np.zeros(NP_SHAPE)

    AddTextToImage(bg_img_final,section_title,True)
    WriteVideoPart(bg_img_final,TITLE_SECTION_DURATION,out)
    return TITLE_SECTION_DURATION

def BuildDataSection(text_data,out,audio_file_path):
    # text to speech
    GenerateAudioFromText(text_data,audio_file_path)
    
    # speech to text + process
    segments, all_text_whisper = ExtractTimestampsWhisper(audio_file_path)
    segments = CleanTimestamps(segments)
    word_time_stamps = CombineSegmentsToWordList(segments)
    keyphrases_in_text = ExtractcKeyphrasesFromText(all_text_whisper)
    word_time_stamps = CombineKeyphrases(keyphrases_in_text, word_time_stamps)
    segments = SplitToSegments(word_time_stamps)
    segments = AddBlanksAndStrech(segments)
    DownloadImagesFromSegments(segments,IMAGES_PATH)

    # build video
    audio = AudioFileClip(audio_file_path)
    audio_dur = audio.duration

    if audio_dur-segments[-1]['end']>0.2:
        segments = segments + [{
            'text':'',
            'start': segments[-1]['end'],
            'end': audio_dur,
            'words':[]
        }]
    segments[-1]['end'] = audio_dur

    for segment in tqdm(segments):
        text = segment['text']
        words = segment['words']
        key_words = [word for word in words if word['key']]
        bg_img_final = np.zeros(NP_SHAPE)

        if len(key_words)>0:
            key_word = key_words[0]['text']
            bg_img = None
            while True:
                try:
                    bg_img_path = SelectImageByKeyphrase(key_word.lower(),IMAGES_PATH)
                except:
                    break
                try:
                    bg_img = cv2.imread(bg_img_path)
                    bg_img.shape
                    break
                except:
                    remove(bg_img_path)
            if bg_img is not None:
                # resize it to fit to VIDEO_SIZE without changing original ratio
                h, w, _ = bg_img.shape
                video_h = VIDEO_SIZE[0]
                video_w = VIDEO_SIZE[1]
                ratio = min(video_w/w, video_h/h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                bg_img = cv2.resize(bg_img, (new_w, new_h))
                
                # place the image in the center of the video
                start_x = (VIDEO_SIZE[1] - new_w) // 2
                start_y = (VIDEO_SIZE[0] - new_h) // 2
                bg_img_final[start_y:start_y+new_h, start_x:start_x+new_w, :] = bg_img

        AddTextToImage(bg_img_final,text)

        # set image duration
        duration = segment['end'] - segment['start']
        WriteVideoPart(bg_img_final,duration,out)
    return audio_dur

def ChapterToVideo(chapter,idx,audios_path,shorts_path):
    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    final_video_path = join(shorts_path,f'{idx+1}.mp4')
    tmp_video_path = join(shorts_path,'tmp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_video_path, fourcc, FPS, VIDEO_SIZE[::-1])

    audios_data = []
    
    title_dur = BuildSectionTitle(chapter['Title'],out)
    audios_data.append({
        'audio_file_path':None,
        'start': 0,
        'duration': title_dur
    })
    last_start = audios_data[-1]['start']
    last_dur = audios_data[-1]['duration']
    for section in chapter['Sections']:
        title_dur = BuildSectionTitle(section['Title'],out)
        audio_file_path = join(audios_path,section['Title'].replace('\n', ' ') + '.mp3')
        text_dur = BuildDataSection(section['Text'],out,audio_file_path)
        audios_data.append({
            'audio_file_path':audio_file_path,
            'start': last_start + last_dur + title_dur,
            'duration': text_dur
        })
        last_start = audios_data[-1]['start']
        last_dur = audios_data[-1]['duration']

    # Release everything after the job is finished
    out.release()
    cv2.destroyAllWindows()

    print("wrote video")

    video = VideoFileClip(tmp_video_path)
    video_duration = video.duration

    audios = []

    last_end = 0
    for audio_data in audios_data:
        if audio_data['audio_file_path'] is None:
            silent_clip = AudioArrayClip(np.zeros((int(audio_data['duration'] * SOUND_HZ), 2)), fps=SOUND_HZ) # SOUND_HZ is a common framerate for audio
            audios.append(silent_clip)
            last_end = audio_data['start'] + audio_data['duration']
            continue

        if audio_data['start'] > last_end:
            silent_duration = audio_data['start'] - last_end
            silent_clip = AudioArrayClip(np.zeros((int(silent_duration * SOUND_HZ), 2)), fps=SOUND_HZ) # SOUND_HZ is a common framerate for audio
            audios.append(silent_clip)
            
        audio = AudioFileClip(audio_data['audio_file_path']).subclip(0, audio_data['duration'])
        audios.append(audio)
        last_end = audio_data['start'] + audio_data['duration']

    if video_duration > last_end:
        silent_duration = video_duration - last_end
        silent_clip = AudioArrayClip(np.zeros((int(silent_duration * SOUND_HZ), 2)), fps=SOUND_HZ)
        audios.append(silent_clip)
    
    print("added audio")

    final_audio = concatenate_audioclips(audios)
    video = video.set_audio(final_audio)
    video.write_videofile(final_video_path, codec='libx264')
    
    return tmp_video_path