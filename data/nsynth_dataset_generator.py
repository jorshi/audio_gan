#!/usr/bin/python
"""
Script for loading subsets of the NSynth dataset

See https://magenta.tensorflow.org/datasets/nsynth#feature-encodings for more information
"""

import os
import sys
import argparse
import json
import librosa
import numpy as np
from tqdm import tqdm


def extract_data_subset(metadata, families, sources, pitches, folder):
    """
    Look through the dataset metadata file and compile a list of audio files to
    include in this dataset.

    Args:
        metadata (dict): a dictionary from the dataset metadata json
        families (list): a list of ids of instrument families to use
        sources (list): a list of ids of instrument sources to use
        folder (str): location of the NSynth folder
        pitches (list): set of pitches to include, if None then all pitches

    Returns:
        list: a list of audio filenames for this subset of NSynth
    """

    audio_files = []
    seen_pitches = {}
    for key in metadata:
        if metadata[key]['instrument_family'] in families and metadata[key]['instrument_source'] in sources\
                and (pitches is None or metadata[key]['pitch'] in pitches):

            filename = "{}.wav".format(key)
            audio_files.append(os.path.join(folder, "audio", filename))

            pitch = metadata[key]['pitch']
            if pitch in seen_pitches:
                seen_pitches[pitch] += 1
            else:
                seen_pitches[pitch] = 1

    return audio_files, seen_pitches


def extract_audio(audio_files, length=16384, sr=16000):
    """
    Loads all the WAV files at specific sampling rate and normalizes

    Args:
        audio_files (list): List of audio filenames to load
        length (int, optional): Length that each sample should be trimmed to. defaults to 16384
        sr (int, optional): Sample rate to load audio at. defaults to 16000

    Returns:

    """

    # Load and trim the audio files to the correct length
    audio_data = np.zeros((len(audio_files), length))
    for i, filename in enumerate(audio_files):
        audio, _ = librosa.load(filename, sr=sr)
        trim = min(len(audio), length)
        audio_data[i, 0:trim] = audio[0:trim]

    # Normalize audio samples
    peak = max(abs(audio_data.max()), abs(audio_data.min()))
    audio_data /= peak

    return audio_data


def extract_melspectrograms(audio_files):
    """
    Loads all audio files and creates melspectrogram images

    Args:
        audio_files (list): list of audio files to load

    Returns:
        np.ndarray: melspectrograms of extracted audio
    """

    audio_data = extract_audio(audio_files, length=16256)
    mel_data = np.zeros((len(audio_data), 128, 128))

    for i in tqdm(range(len(audio_data))):
        S = librosa.feature.melspectrogram(audio_data[i], sr=16000, n_fft=2056, hop_length=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        mel_data[i] = (S_dB + 80.0) / 80.0

    return mel_data


def main(arguments):
    """
    Script entry point

    Args:
        arguments: command line arguments
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('nsynth_folder', help="Location of the NSynth dataset", type=str)
    parser.add_argument('output', help="Filename for numpy dataset output", type=str)
    parser.add_argument('-f', '--family', nargs='+', help="List of instrument family ids", default=[], type=int)
    parser.add_argument('-s', '--source', nargs='+', help="List of instrument source ids", default=[], type=int)
    parser.add_argument('-p', '--pitch', nargs='+', help="List of pitches", default=None, type=int)
    parser.add_argument('-m', '--mel', dest='extraction', action='store_const', const='mel', default='time')
    parser.add_argument('--note_freq', action='store_const', const=True, default=False)

    args = parser.parse_args(arguments)

    if not len(args.family):
        raise Exception("Must supply at least one instrument family id using -f/--family argument. "
                        "See https://magenta.tensorflow.org/datasets/nsynth#feature-encodings for id info.")

    if not len(args.source):
        raise Exception("Must supply at least one instrument source id using -s/--source argument. "
                        "See https://magenta.tensorflow.org/datasets/nsynth#feature-encodings for id info.")

    with open(os.path.join(args.nsynth_folder, "examples.json")) as fp:
        metadata = json.load(fp)

    print("Compiling audio files from NSynth metadata")
    audio_files, pitches = extract_data_subset(metadata, args.family, args.source, args.pitch, args.nsynth_folder)

    if args.note_freq:
        sorted_pitches = sorted(pitches.items(), key=lambda x: x[1])
        print(sorted_pitches)
        return

    if args.extraction == 'mel':
        print('Extracting audio files and creating mel spectrograms')
        data = extract_melspectrograms(audio_files)

    else:
        print("Extracting audio files")
        data = extract_audio(audio_files)

    np.save(args.output, data)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
