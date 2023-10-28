from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip, TextClip, ColorClip
from skimage.filters import gaussian

VIDEO_SIZE = (1080,1920)

def BlurVideo(video, sigma=6):
    def blur(image):
        return gaussian(image.astype(float), sigma=sigma)
    return video.fl_image( blur )

def FitClip(video_clip,new_dimension):
    original_width = video_clip.size[0]
    original_height = video_clip.size[1]
    original_ratio = original_width / original_height

    new_width = new_dimension[0]
    new_height = new_dimension[1]
    new_ratio = new_width / new_height

    width_ratio = new_width/original_width
    height_ratio = new_height/original_height

    if new_ratio > original_ratio:
        resize_width = new_width
        resize_height = int(new_width / original_ratio)
    else:
        resize_width = int(new_height * original_ratio)
        resize_height = new_height

    resized_clip = video_clip.resize((resize_width, resize_height))

    
    if height_ratio>width_ratio:
        crop_x = (resize_width - new_width) // 2
        cropped_clip = resized_clip.crop(x1=crop_x, x2=crop_x+new_width)
    else:
        crop_y = (resize_height - new_height) // 2
        cropped_clip = resized_clip.crop(y1=crop_y, y2=crop_y+new_height)
    return cropped_clip

def ResizeVideo(clip, new_dimension):
    original_width = clip.size[0]
    original_height = clip.size[1]
    original_ratio = original_width / original_height

    new_width = new_dimension[0]
    new_height = new_dimension[1]
    new_ratio = new_width / new_height

    if new_ratio > original_ratio:
        resize_width = int(new_height * original_ratio)
        resize_height = new_height
    else:
        resize_width = new_width
        resize_height = int(new_width / original_ratio)

    resized_clip = clip.resize((resize_width, resize_height)).set_position('center')
    return resized_clip

def CenteredWithBlurred(clip,new_dimension):
    resized_clip = ResizeVideo(clip, new_dimension)
    bg_clip = BlurVideo(clip)
    bg_clip = FitClip(bg_clip,new_dimension).set_pos('center')

    # Overlay the resized clip onto the blank video at the center position
    final_clip = CompositeVideoClip([bg_clip,resized_clip])

    return final_clip