

import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
import emoji
from pathlib import Path
import warnings
import unicodedata

from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.util import ngrams

nltk.download('stopwords')
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
def clean_text(s):
    """Clean a single text entry."""
    if not isinstance(s, str):
        return ""
    # fix encoding weirdness and accents
    s = unicodedata.normalize("NFKC", s)
    # lowercase
    s = s.lower()
    # remove URLs
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    # remove mentions and hashtags
    s = re.sub(r"@\w+|#\w+", " ", s)
    # keep letters (English + Spanish + German)
    s = re.sub(r"[^a-zA-ZáéíóúñüäößÄÖÜ ]", " ", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    s = emoji.replace_emoji(s, replace='')
    return s

def preprocess_file(input_path, output_path):
    """Load, clean, and save CSV."""
    print(f"\nProcessing {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8")

    # detect text column
    possible = [c for c in df.columns if c.lower() in ("text")]
    text_col = possible[0] if possible else df.columns[0]

    # keep text + label if available
    cols = ['id',text_col]
    if 'polarization' in df.columns:
        cols.append('polarization')
    df = df[[c for c in cols if c in df.columns]]

    # clean text
    df['text'] = df[text_col].apply(clean_text)

    # drop blanks + duplicates
    df = df[df['text'].str.strip() != ""]
    df = df.drop_duplicates(subset=['text'])

    # save cleaned file
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved cleaned file: {output_path} ({len(df)} rows)")

# run for all three
files = [
    ("data/eng.csv", "data/eng_clean.csv"),
    ("data/deu.csv", "data/deu_clean.csv"),
    ("data/spa.csv", "data/spa_clean.csv")
]

for inp, out in files:
    preprocess_file(inp, out)
