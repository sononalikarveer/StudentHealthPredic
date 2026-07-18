import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
MODELS_DIR = OUTPUTS_DIR / 'models'
OOF_DIR = OUTPUTS_DIR / 'oof'
SUBMISSIONS_DIR = OUTPUTS_DIR / 'submissions'
REPORTS_DIR = OUTPUTS_DIR / 'reports'

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
SAMPLE_SUB_PATH = DATA_DIR / 'sample_submission.csv'

SEED = 42
N_FOLDS = 5
