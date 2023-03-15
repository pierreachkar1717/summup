""" 
    A module that contains a class that summarizes arxiv papers.
"""

from pdf_summarizer import PDFSummarizer
import requests
from bs4 import BeautifulSoup
import io
from dotenv import load_dotenv
import openai
import os


class arxiv_summarizer(PDFSummarizer):
    """
    A class that summarizes arxiv papers.
    """

    def __init__(self, arxiv_paper_link, method):
        self.arxiv_paper_link = arxiv_paper_link
        self.method = method
        self.text = self.get_arxiv_text(arxiv_paper_link)
        self.chunks = self.preprocess_and_chunk_text(self.text)

    @staticmethod
    def get_arxiv_text(link):
        """
        Retrieve the text of a arxiv paper using its link.


        """
        # access the arxiv link and and get the pdf file
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")

        # get the pdf link with a specific class
        pdf_link = soup.find("a", {"class": "mobile-submission-download"}).get("href")

        # add https://arxiv.org/ to the link
        pdf_link = "https://arxiv.org/" + pdf_link

        # get the pdf file
        response_2 = requests.get(pdf_link)
        pdf_bytes = io.BytesIO(response_2.content)

        # use get_pdf_text function from pdf_summarizer.py
        content = PDFSummarizer.get_pdf_text(pdf_bytes)

        return content

    # override the _summarize_with_gpt method from pdf_summarizer.py
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
                        "content": "You are an AI research assitant. Your task is to summarise scientific papers clearly, comprehensively and in an understandable way. The tone of the summary should be formal and scientific.",
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

    def summarize(self):
        """
        Summarize the input text using the specified summarization method and export the output to text file.

        Returns:
            A string containing the summarized text.
        """
        if self.method == "openai":
            summary = self._summarize_with_gpt()
            with open("summary.txt", "w") as f:
                f.write(summary)
            return summary
        else:
            raise ValueError("Invalid summarization method.")
