import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict

# Download necessary NLTK data (run this once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError: # Changed from nltk.downloader.DownloadError
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Changed from nltk.downloader.DownloadError
    nltk.download('punkt')
# Add download for 'punkt_tab' as it's explicitly requested by the error
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError: # Changed from nltk.downloader.DownloadError
    nltk.download('punkt_tab')


def summarize_text(text, num_sentences=3):
    """
    Summarizes the given text by extracting the most important sentences.

    Args:
        text (str): The input text to be summarized.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: A concise summary of the input text.
    """
    # 1. Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # If the text has fewer sentences than requested, return the whole text
    if len(sentences) <= num_sentences:
        return text

    # 2. Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # 3. Calculate word frequencies
    word_frequencies = defaultdict(int)
    for word in filtered_words:
        word_frequencies[word] += 1

    # 4. Calculate maximum frequency to normalize
    if not word_frequencies: # Handle empty word_frequencies case
        return "Cannot summarize: No meaningful words found."
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # 5. Score sentences based on word frequencies
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]

    # 6. Select the top-scoring sentences
    # Sort sentences by score in descending order and get the indices
    sorted_sentence_indices = sorted(sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True)

    # Get the indices of the top N sentences, ensuring they are in their original order
    top_sentence_indices = sorted(sorted_sentence_indices[:num_sentences])

    # Construct the summary
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    summary = ' '.join(summary_sentences)

    return summary

# --- Example Usage ---
if __name__ == "__main__":
    article = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on enabling computers to understand, interpret, and generate human language. It combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models. NLP has made significant strides in recent years, largely due to advancements in deep learning architectures like transformers.

    Key applications of NLP include sentiment analysis, machine translation, spam detection, chatbots, and text summarization. Sentiment analysis, for instance, helps businesses understand customer opinions from reviews and social media. Machine translation, like Google Translate, allows for communication across language barriers. Chatbots are increasingly used for customer service and information retrieval.

    Text summarization, the focus of this tool, aims to create a concise and coherent summary of a longer text document while retaining the main points. There are two main types: extractive and abstractive. Extractive summarization identifies and pulls key sentences or phrases directly from the original text. Abstractive summarization, on the other hand, generates new sentences that capture the essence of the original text, often requiring more advanced AI models.

    Challenges in NLP include ambiguity in language, variations in grammar, and the need for vast amounts of data for training models. However, ongoing research continues to push the boundaries of what's possible, leading to more sophisticated and accurate NLP systems. The future of NLP holds immense potential for transforming how humans interact with technology and process information.
    """

    print("Original Article:")
    print(article)
    print("\n" + "="*50 + "\n")

    # Summarize to 3 sentences (default)
    summary_3_sentences = summarize_text(article, num_sentences=3)
    print("Summary:") # Removed "(3 sentences):"
    print(summary_3_sentences)
    print("\n" + "="*50 + "\n")
