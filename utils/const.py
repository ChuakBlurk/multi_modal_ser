import os

PROCESSED_DATA_PATH = "E:/datasets/preprocessed"

TRAIN_CSV_PATH = os.path.join(PROCESSED_DATA_PATH, "training_data.csv")
EVAL_CSV_PATH = os.path.join(PROCESSED_DATA_PATH, "evaluation_data.csv")
TEST_CSV_PATH = os.path.join(PROCESSED_DATA_PATH, "prediction_data.csv")

EXTRACTIONMAP_PATH = os.path.join(PROCESSED_DATA_PATH, "extractionmap")
FACE_PATH = os.path.join(PROCESSED_DATA_PATH, "face")
SPECTROGRAM_PATH = os.path.join(PROCESSED_DATA_PATH, "spectrogram_concat")