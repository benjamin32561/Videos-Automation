import whisper_timestamped as whisper
from torch import cuda

DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
WHISPER_SIZE = "medium.en"

def CleanTimestamps(whisper_result):
  final = []
  for segment in whisper_result:
    words = []
    for word in segment['words']:
      words.append({
          'text':word['text'],
          'start':word['start'],
          'end':word['end'],
      })
    final.append({
        'text':segment['text'],
        'start':segment['start'],
        'end':segment['end'],
        'words':words,
    })
  return final
    
def ExtractTimestampsWhisper(audio_path):
  audio = whisper.load_audio(audio_path)
  model = whisper.load_model(WHISPER_SIZE, device=DEVICE)

  res = whisper.transcribe(model, audio, language="en")
  data, all_text = res['segments'], res['text']

  return data, all_text