# Run source separation
import subprocess

subprocess.run([
    'python', '../sound-separation/models/tools/process_wav.py',
    '--model_dir', '../sound-separation/models/bird_mixit/bird_mixit_model_checkpoints/output_sources8',
    '--checkpoint', '../sound-separation/models/bird_mixit/bird_mixit_model_checkpoints/output_sources8/model.ckpt-2178900',
    '--num_sources', '8',
    '--input', '../sound-separation/models/bird_mixit/chorus.wav',
    '--output', 'annotation/_output/test/chorus_source.wav'
])
