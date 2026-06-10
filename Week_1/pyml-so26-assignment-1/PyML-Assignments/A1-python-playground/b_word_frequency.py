"""
Fill out the code below so that each function behaves as described in its
docstring. Together they form a pipeline that reads text, analyzes word
frequencies, and writes a formatted report.

DO NOT MODIFY THE FUNCTION SIGNATURES OR THE STOPWORDS SET.
"""

import os


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "so",
    "yet",
    "for",
    "nor",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "up",
    "as",
    "is",
    "it",
    "its",
    "be",
    "was",
    "are",
    "were",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "his",
    "her",
    "they",
    "them",
    "their",
    "this",
    "that",
    "with",
    "from",
    "not",
    "no",
    "if",
    "then",
    "than",
    "also",
    "just",
    "more",
    "into",
}


def normalize(text: str) -> list[str]:
    """
    Cleans raw text into a list of words suitable for frequency analysis.

    Strips leading and trailing whitespace from the full text, lowercases
    everything, and splits on whitespace. Filters out any token that either
    contains no alphabetic characters or appears in STOPWORDS.

    Arguments:
        text -- a raw string of text

    Returns:
        list[str] -- a list of clean, lowercase, content-bearing words
    """

    # ------ SOLUTION GOES HERE!  ------
    # Strip, lowercase, split
    tokens = text.strip().lower().split()

    clean_words = []
    for tok in tokens:
        # Debe contener al menos una letra
        if not any(ch.isalpha() for ch in tok):
            continue

        # Quitar puntuación simple en bordes
        tok = tok.strip(".,!?;:\"'()[]{}")

        # Filtrar stopwords
        if tok in STOPWORDS:
            continue

        clean_words.append(tok)

    return clean_words


def word_frequencies(words: list[str]) -> dict[str, int]:
    """
    Counts the occurrences of each word and returns results sorted by
    frequency (descending), with ties broken alphabetically (ascending).

    Arguments:
        words -- a list of clean words, as returned by normalize()

    Returns:
        dict[str, int] -- a dictionary mapping each word to its count,
            ordered by frequency descending, then alphabetically ascending
    """

    # ------ SOLUTION GOES HERE!  ------
    # Contar
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    # Ordenar:
    # 1) frecuencia descendente → -count
    # 2) alfabético ascendente → word
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    # Regresar como dict ordenado
    return dict(sorted_items)


def format_entry(rank: int, word: str, count: int, total: int) -> str:
    """
    Formats a single line of the frequency report using an f-string.

    The line must follow this exact layout:
        - Rank: right-aligned in a field of width 4, followed by a period
        - Word: left-aligned in a field of width 20
        - Count: right-aligned in a field of width 6
        - Percentage of total words: right-aligned in a field of width 8,
          shown to two decimal places, followed by a percent sign

    Example output (rank=1, word="python", count=42, total=200):
        '   1. python                  42   21.00%'

    Arguments:
        rank  -- the 1-based rank of this word by frequency
        word  -- the word being reported
        count -- the number of times the word appears
        total -- the total number of words (after normalization)

    Returns:
        str -- a single formatted report line

    (_Hint: See Python's Format Specification Mini-Language for alignment and
    width syntax: https://docs.python.org/3/library/string.html#format-specification-mini-language_)
    """

    # ------ SOLUTION GOES HERE!  ------
    percentage = (count / total) * 100

    return f"{rank:>4}. {word:<20}{count:>6}{percentage:>8.2f}%"

    """
    Orchestrates the full pipeline and writes a frequency report to a file.

    Determines whether `source` is a filepath or a raw string: if a file
    exists at that path, read the text content from it; otherwise treat
    `source` itself as the text. (_Hint: `os.path.exists(source)` is the
    right tool for this check._) Creates any intermediate output directories
    if they do not already exist. Writes the top_n most frequent words to
    the file at output_path, one formatted line per word, using format_entry().
    Wraps all file operations in a try/except and raises an IOError with a
    descriptive message if anything goes wrong.

    Arguments:
        source      -- a filepath to a .txt file, or a raw string of text
        output_path -- the filepath where the report will be written
        top_n       -- the number of top words to include (default: 10)

    Returns:
        None

    Raises:
        IOError -- if reading from source or writing to output_path fails
    """


def write_report(source: str, output_path: str, top_n: int = 10) -> None:
    try:
        # Leer texto
        if os.path.exists(source):
            with open(source, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = source

        # Pipeline
        words = normalize(text)
        freqs = word_frequencies(words)
        total = sum(freqs.values())

        # Crear directorios si no existen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Escribir reporte
        with open(output_path, "w", encoding="utf-8") as f:
            for rank, (word, count) in enumerate(list(freqs.items())[:top_n], start=1):
                f.write(format_entry(rank, word, count, total) + "\n")

    except Exception as e:
        raise IOError(f"Error processing report: {e}")

    # ------ SOLUTION GOES HERE!  ------


if __name__ == "__main__":
    sample_text = """
    To be or not to be that is the question whether tis nobler in the mind
    to suffer the slings and arrows of outrageous fortune or to take arms
    against a sea of troubles and by opposing end them to die to sleep no
    more and by a sleep to say we end the heartache and the thousand natural
    shocks that flesh is heir to tis a consummation devoutly to be wished
    """

    print("Running on raw string...")
    write_report(sample_text, "output/raw_report.txt", top_n=10)
    print("Report written to output/raw_report.txt")

    print("Running on file...")
    write_report("sample.txt", "output/file_report.txt", top_n=10)
    print("Report written to output/file_report.txt")
