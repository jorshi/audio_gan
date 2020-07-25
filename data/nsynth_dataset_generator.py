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


def extract_data_subset(metadata, families, sources, folder):
    """
    Look through the dataset metadata file and compile a list of audio files to
    include in this dataset.

    Args:
        metadata (dict): a dictionary from the dataset metadata json
        families (list): a list of ids of instrument families to use
        sources (list): a list of ids of instrument sources to use
        folder (str): location of the NSynth folder

    Returns:
        list: a list of audio filenames for this subset of NSynth
    """

    audio_files = []
    for key in metadata:
        if metadata[key]['instrument_family'] in families and metadata[key]['instrument_source'] in sources:
            filename = "{}.wav".format(key)
            audio_files.append(os.path.join(folder, "audio", filename))

    return audio_files


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
    audio_files = extract_data_subset(metadata, args.family, args.source, args.nsynth_folder)

    print("Extracting audio files")
    data = extract_audio(audio_files)

    np.save(args.output, data)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
