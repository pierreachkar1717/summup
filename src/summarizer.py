"""
A module that contains a class that summarizes a given text using a specified summarization method. 
"""

import re
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import os
import openai
from nltk.corpus import stopwords
import nltk

class Summarizer:
    """
    A class that summarizes a given text using a specified summarization method.

    Attributes:
        text (str): The input text.
        method (str): The summarization method.
        chunks (list of str): A list of preprocessed text chunks, where each chunk
            contains only full sentences.

    Methods:
        preprocess_and_chunk_transcript(text, chunk_size=2500):
            Preprocess a raw text by removing unnecessary tokens, tokenize it into sentences,
            and divide it into chunks. Each chunk contains only full sentences.
    """

    def __init__(self, text, method):
        """
        Initialize the summarizer object.

        Args:
            text (str): The input text to be summarized.
            method (str): The summarization method to be used.
        """
        self.text = text
        self.method = method
        self.chunks = self.preprocess_and_chunk_text(text)

    @staticmethod
    def preprocess_and_chunk_text(text, chunk_size=3000):
        """
        Preprocess a raw text by removing unnecessary tokens,
        tokenize it into sentences, and divide it into chunks.
        Each chunk contains only full sentences.

        Args:
            text (str): The raw transcript text.

        Returns:
            A list of strings, where each string represents a chunk of the transcript.
            Each chunk has a length of at most chunk_size tokens, and contains only full sentences.
        """

        stop_words = set(stopwords.words("english"))

        # Preprocess the transcript by removing unnecessary tokens and stopwords
        cleaned_transcript = re.sub(
            r"\[.*?\]", "", text
        )  # remove square brackets and their contents
        cleaned_transcript = re.sub(
            r"\(.*?\)", "", cleaned_transcript
        )  # remove parentheses and their contents
        cleaned_transcript = re.sub(
            r"<.*?>", "", cleaned_transcript
        )  # remove angle brackets and their contents
        cleaned_transcript = re.sub(
            r"\n+", " ", cleaned_transcript
        )  # replace multiple line breaks with a single space
        cleaned_transcript = re.sub(
            r"[^\w\s\.\?\!]", "", cleaned_transcript
        )  # remove all non-word, non-space, non-punctuation characters
        cleaned_transcript = re.sub(r"[^\x00-\x7F]+", "", cleaned_transcript)
        cleaned_transcript = " ".join(
            [word for word in cleaned_transcript.split() if word not in stop_words]
        )

        # Tokenize the transcript into sentences
        sentences = sent_tokenize(cleaned_transcript)

        # Divide the transcript into chunks
        MAX_CHUNK_LENGTH = chunk_size
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= MAX_CHUNK_LENGTH:
                # Add sentence to current chunk
                current_chunk += sentence + " "
            else:
                # Start a new chunk with the current sentence
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # remove empty chunks
        chunks = [chunk for chunk in chunks if chunk]

        return chunks

    def _summarize_with_gpt(self):
        """
        Summarize the input text using OpenAI's GPT-3.5 Turbo API.

        Returns:
            A string containing the summarized text.
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        full_summary = list()

        for chunk in self.chunks:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI summarizer tool. Your mission is to help summarize Youtube videos.",
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following text: {chunk}",
                    },
                ],
            )

            content = completion["choices"][0]["message"]["content"]

            full_summary.append(content)

        # convert list to string
        full_summary = " ".join(full_summary)

        return full_summary

    def summarize(self):
        """
        Summarize the input text using the specified summarization method.

        Returns:
            A string containing the summarized text.
        """
        if self.method == "openai":
            return self._summarize_with_gpt()
        else:
            raise ValueError("Invalid summarization method.")

    def summarize_in_bullets(self):
        """
        Summarize the input text using the specified summarization method,
        and format the output as a list of bullet points.

        Returns:
            A string containing the summarized text, formatted as a list of bullet points.
        """
        summary = self.summarize()
        bullet_points = summary.split(".")
        bullet_points = [
            f"* {point.strip()}" for point in bullet_points if point.strip()
        ]
        return "\n".join(bullet_points)

# TODO: Add more summarization methods from hugging face transformers
