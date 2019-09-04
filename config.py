import json
from pathlib import Path


HOMEDIR = Path('.').resolve()
BUILDDIR = HOMEDIR / 'build'
DATADIR  = HOMEDIR / 'data'

GCP_REGION = 'europe-west3'
GCP_CLUSTER = 'case01'
