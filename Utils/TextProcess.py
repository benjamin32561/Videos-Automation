import random
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np
from google.cloud import texttospeech

N_WORDS_PER_SEGMENT_RANGE = (2,5)

MAX_SILENCE_BEFORE_SPLIT = 3 # seconds
MAX_SHORT_LENGTH_RANGE = (60,140) # seconds
MIN_SHORT_LENGTH = 10 # seconds
MAX_WORDS_PER_SHORT = 10000
SPLIT_SHORT_CHARACTERS = ['.', '?', '!']
MAX_GAP_IN_SEGMENT = 0.1 # seconds

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

def SplitToSegments(words):
    lines = []
    while len(words)>0:
        n_words = -1
        for idx, word in enumerate(words[:N_WORDS_PER_SEGMENT_RANGE[1]]):
          if word['text'][-1] in SPLIT_SHORT_CHARACTERS:
            n_words = idx+1
            break
        if n_words==-1 or N_WORDS_PER_SEGMENT_RANGE[1]<n_words:
          n_words=N_WORDS_PER_SEGMENT_RANGE[1]
        if N_WORDS_PER_SEGMENT_RANGE[0]<n_words:
          n_words = random.randint(N_WORDS_PER_SEGMENT_RANGE[0], n_words)
        if len(words) <= n_words:
            line = words
            words = []
        else:
            line = []
            line.append(words[0])
            words.pop(0)
            while line[-1]['end']-words[0]['start']<MAX_GAP_IN_SEGMENT and len(line)<n_words:
              line.append(words[0])
              words.pop(0)

            # line = words[:n_words]
            # words = words[n_words:]
        
        line_dict = {
            'text': ' '.join([w['text'] for w in line]),
            'start': line[0]['start'],
            'end': line[-1]['end'],
            'words': line
        }
        lines.append(line_dict)
    return lines

def CombineSegmentsToWordList(segments:list):
    to_ret = []
    for segments in segments:
        to_ret+=segments['words']
    return to_ret

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
  def __init__(self, model, *args, **kwargs):
    super().__init__(
      model=AutoModelForTokenClassification.from_pretrained(model),
      tokenizer=AutoTokenizer.from_pretrained(model),
      *args,
      **kwargs
    )

  def postprocess(self, all_outputs):
    results = super().postprocess(
        all_outputs=all_outputs,
        aggregation_strategy=AggregationStrategy.SIMPLE,
    )
    return np.unique([result.get("word").strip().lower() for result in results])
  
def ExtractcKeyphrasesFromText(text:str):
  model_name = "ml6team/keyphrase-extraction-kbir-kpcrowd"
  extractor = KeyphraseExtractionPipeline(model=model_name)

  keyphrases = extractor(text).tolist()

  return keyphrases

def CombineKeyphrases(keyphrases:list, words):
  for keyphrase in keyphrases:
    keyphrase = keyphrase.lower()
    if keyphrase.count(' ')==0:
      for idx in range(len(words)):
        if keyphrase == words[idx]['text'].lower():
          words[idx]['key']=True
        elif 'key' not in words[idx].keys():
          words[idx]['key']=False
    else:
      parts = keyphrase.split(' ')
      n_parts = len(parts)
      n_words = len(words)-n_parts
      idx = 0
      while idx < n_words:
        if words[idx]['text'].lower()!=parts[0]:
          idx+=1
          continue
        current = ' '.join([word['text'] for word in words[idx:idx+n_parts]])
        cleaned = ''.join([char for char in current if char.isalpha() or char.isspace()])
        if keyphrase==cleaned.lower():
          new_word = {
              'text': current,
              'start': words[idx]['start'],
              'end': words[idx+n_parts]['end'],
              'key': True
          }
          words[idx:idx+n_parts] = [new_word]
          n_words-=n_parts
        idx+=1
  return words

def GenerateEmptyShort(start=-1,end=0,segments=[]):
  dur = 0
  if start!=-1:
    dur = end-start
  else:
    start = 0
  return {
    'start':start,
    'end':end,
    'duration':dur,
    'segments':segments
  }

def SplitToShorts(data:list):
  shorts = []
  short = GenerateEmptyShort()
  last_end = -1
  max_short_length = random.randint(MAX_SHORT_LENGTH_RANGE[0],MAX_SHORT_LENGTH_RANGE[1])
  for line in data:
    if short['duration']<max_short_length \
        and last_end!=-1 \
        and line['start']-last_end<MAX_SILENCE_BEFORE_SPLIT:
      if short['start']==-1:
        short['start'] = max(line['start']-0.1,0)
      short['segments'].append(line)
      short['end']=line['end']
      short['duration']=short['end']-short['start']
    else:
      # add short to shorts list
      shorts.append(short)
      short=GenerateEmptyShort(max(line['start']-0.1,0),line['end'],[line])
    last_end = line['end']
  if len(shorts)>1:
    shorts = shorts[1:]
  return [s for s in shorts if s['duration']>MIN_SHORT_LENGTH]

def AddBlanksAndStrech(segments:list):
  if len(segments)==0:
    return segments
  if segments[0]['start']>0.1:
        segments = [{
            'text':'',
            'start': 0.0,
            'end': segments[0]['start'],
            'words':[]
        }]+segments
  else:
    segments[0]['start'] = 0.0

  idx = 0
  while idx<len(segments)-1:
    if segments[idx+1]['start']-segments[idx]['end']>1:
      to_add = {
        'text':'',
        'start': segments[idx]['end']+0.2,
        'end': segments[idx+1]['start'],
        'words':[]
      }
      segments[idx]['end']+=0.2
      segments = segments[:idx+1]+[to_add]+segments[idx+1:]
      idx+=1
    else:
      segments[idx]['end'] = segments[idx+1]['start']
    idx+=1
  return segments