# Grammar-Scoring-Engine
# Speech Grammar Scoring Engine

An ML system that predicts the grammar proficiency score of spoken English audio on a scale of 1 to 5.

## Overview

The system takes a `.wav` audio file, converts speech to text using Whisper, extracts audio, text, and grammar features, and predicts a grammar score using a Random Forest regression model.

## Pipeline

Audio → Whisper Transcription → Feature Extraction → Random Forest → Grammar Score

## Features

Audio features using Librosa

Text features such as word count, sentence length and vocabulary usage

Grammar features using spaCy POS tagging and dependency analysis

## Model

Random Forest Regressor

Evaluation metric: RMSE

Baseline RMSE: 1.2382

5 Fold CV RMSE: 0.7876

## Tech Stack

Python, Whisper, Librosa, spaCy, Pandas, NumPy, Scikit learn

##Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## Dataset

769 training samples and 216 test samples of spoken English audio.

## Author

Rijin Shaji
