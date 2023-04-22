import os
import ffmpeg
from pytube import YouTube


# code source from https://github.com/deepimagination/TalkingHead-1KH/
def download_video(dataset_directory, video_id):
    video_path = f'{dataset_directory}/{video_id}.mp4'
    if not os.path.isfile(video_path):
        try:
            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            stream = yt.streams.filter(subtype='mp4', only_video=True, adaptive=True).first()
            if stream is None:
                stream = yt.streams.filter(subtype='mp4').first()
            stream.download(output_path=dataset_directory, filename=video_id + '.mp4')
        except Exception as e:
            print(e)
            print(f'Failed to download {video_id}')
    else:
        print(f'File exists: {video_id}')


def get_height_and_width(file_path):
    probe = ffmpeg.probe(file_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    height = int(video_stream['height'])
    width = int(video_stream['width'])

    return height, width


def trim_and_crop(clip_dir, output_dir, clip_params):
    video_name, H, W, S, E, L, T, R, B = clip_params.strip().split(',')
    H, W, S, E, L, T, R, B = int(H), int(W), int(S), int(E), int(L), int(T), int(R), int(B)
    output_filename = f'{video_name}_S{S}_E{E}_L{L}_T{T}_R{R}_B{B}.mp4'
    output_filepath = os.path.join(output_dir, output_filename)
    if os.path.exists(output_filepath):
        print(f'Output file {output_filepath} exists, skipping.')
        return

    input_filepath = os.path.join(clip_dir, video_name + '.mp4')
    if not os.path.exists(input_filepath):
        print(f'Input file path {input_filepath} does not exist, skipping.')
        return

    h, w = get_height_and_width(input_filepath)
    t = int(T / H * h)
    b = int(B / H * h)
    l = int(L / W * w)
    r = int(R / W * w)

    stream = ffmpeg.input(input_filepath)
    stream = ffmpeg.trim(stream, start_frame=S, end_frame=E+1)
    stream = ffmpeg.crop(stream, l, t, r-l, b-t)
    stream = ffmpeg.output(stream, output_filepath)
    ffmpeg.run(stream)


