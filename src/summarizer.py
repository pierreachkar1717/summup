"""
A module that contains a class that summarizes a given text using a specified summarization method. 
"""

import re
from dotenv import load_dotenv
import os
import openai
from nltk.corpus import stopwords
import nltk
from transformers import pipeline


class Summarizer:
    """
    A class that summarizes a given text using a specified summarization method.

    Attributes:
        text (str): The input text.
        method (str): The summarization method.
        chunks (list of str): A list of preprocessed text chunks, where each chunk
            contains only full sentences.

    Methods:
        preprocess_and_chunk_text(text, chunk_size=2500):
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
        self.chunks = self.preprocess_and_chunk_text(self.text, self.method)

    @staticmethod
    def preprocess_and_chunk_text(text, method):
        """
        Preprocess a raw text by removing unnecessary tokens,
        tokenize it into sentences, and divide it into chunks.
        Each chunk contains only full sentences.

        Args:
            text (str): The raw text text.
            method (str): The summarization method to be used.

        Returns:
            A list of strings, where each string represents a chunk of the text.
            Each chunk has a length of at most chunk_size tokens, and contains only full sentences.
        """

        stop_words = set(stopwords.words("english"))

        # Preprocess the text by removing unnecessary tokens and stopwords
        cleanded_text = re.sub(
            r"\[.*?\]", "", text
        )  # remove square brackets and their contents
        cleanded_text = re.sub(
            r"\(.*?\)", "", cleanded_text
        )  # remove parentheses and their contents
        cleanded_text = re.sub(
            r"<.*?>", "", cleanded_text
        )  # remove angle brackets and their contents
        cleanded_text = re.sub(
            r"\n+", " ", cleanded_text
        )  # replace multiple line breaks with a single space
        cleanded_text = re.sub(
            r"[^\w\s\.\?\!]", "", cleanded_text
        )  # remove all non-word, non-space, non-punctuation characters
        cleanded_text = re.sub(r"[^\x00-\x7F]+", "", cleanded_text)
        cleanded_text = " ".join(
            [word for word in cleanded_text.split() if word not in stop_words]
        )

        # split the text into sentences
        sentences = cleanded_text.split(".")

        # Divide the text into chunks
        if method == "transformers":
            MAX_CHUNK_LENGTH = 512
        elif method == "gpt":
            MAX_CHUNK_LENGTH = 3000

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence.split()) + 1 <= MAX_CHUNK_LENGTH:
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
        # check if the OpenAI API key is set
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise Exception(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        full_summary = list()

        for chunk in self.chunks:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI summariser. Your task is to summarise a given text clearly, comprehensively and in an understandable way.",
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following text in plain language: {chunk}",
                    },
                ],
            )

            content = completion["choices"][0]["message"]["content"]

            full_summary.append(content)

        # convert list to string
        full_summary = " ".join(full_summary)

        return full_summary

    def _summarize_with_transformers(self):
        """
        Summarize the input text using hugging face transformers.

        Returns:
            A string containing the summarized text.
        """
        summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        full_summary = list()
        for chunk in self.chunks:
            summary = summarizer(chunk)
            full_summary.append(summary[0]["summary_text"])
        full_summary = " ".join(full_summary)
        return full_summary

    def summarize(self):
        """
        Summarize the input text using the specified summarization method.

        Returns:
            A string containing the summarized text.
        """
        if self.method == "gpt":
            summary = self._summarize_with_gpt()
            return summary
        elif self.method == "transformers":
            summary = self._summarize_with_transformers()
            return summary

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


# TODO: Add getpass to ask for API key
