# TEXT-SUMMARIZATION-TOOL

**COMPANY NAME** : CODTECH IT SOLUTIONS PVT.LTD
**INTERN NAME**  : PULI MANOHAR ADITYA
**INTERN ID**    : CT08DM486
**DOMAIN**       :ARTIFICIAL INTELLIGENCE
**DURATION**    : 8 WEEKS
**MENTOR**      : NEELA SANTHOSH

DESCRIPTION:
The **Text Summarization Tool** is a Python script designed to automatically generate concise summaries of longer articles or documents. It employs an **extractive summarization** approach, meaning it identifies and extracts the most important sentences directly from the original text to form the summary.

**Here's an overview of its core functionality:**

1.  **Text Preprocessing:**
    * The input text is first tokenized into individual sentences.
    * Words within these sentences are then tokenized, and common "stopwords" (e.g., "the", "is", "a") are removed as they typically don't contribute much to the meaning.

2.  **Word Frequency Analysis:**
    * The script calculates the frequency of each meaningful word remaining after stopword removal.
    * These frequencies are then normalized to give a relative importance score to each word.

3.  **Sentence Scoring:**
    * Each sentence in the original text is assigned a score. This score is determined by summing the normalized frequencies of the important words contained within that sentence. Sentences that contain more high-frequency (and thus, more important) words receive higher scores.

4.  **Summary Generation:**
    * Finally, the sentences are sorted based on their calculated scores in descending order.
    * The top `N` highest-scoring sentences (where `N` is the desired number of sentences in the summary, defaulting to 3) are selected.
    * These selected sentences are then reassembled in their original order to form the final, concise summary.

This tool is particularly useful for quickly understanding the main points of lengthy content, saving time by providing a condensed version of the original material.
