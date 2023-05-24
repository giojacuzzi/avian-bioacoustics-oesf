library(av)

path = normalizePath('~/Desktop/oesf-examples/03.wav', mustWork=T) # 1 hour long file

# Read metadata with av
audio_info = av_media_info(path)$audio
duration = av_media_info(path)$duration
metadata = list(
  channels    = audio_info$channels,
  sample_rate = audio_info$sample_rate,
  codec       = audio_info$codec,
  bitrate     = audio_info$bitrate,
  duration    = duration,
  samples     = duration * audio_info$sample_rate
)

# TODO: check other file diagnostics