""" 
    a module to summarize a given pdf file using a specified summarization method.
"""

from summarizer import Summarizer
from pypdf import PdfReader


class PDFSummarizer(Summarizer):
    """
    A class that summarizes a given PDF file using a specified summarization method.

    Attributes:
        path (str): The path of the PDF file.
        text (str): The text of the PDF file.
        method (str): The summarization method.
        chunks (list of str): A list of preprocessed text chunks, where each chunk
        contains only full sentences.
    """

    def __init__(self, path, method):
        self.path = path
        self.text = self.get_pdf_text(path)
        self.method = method
        self.chunks = self.preprocess_and_chunk_text(self.text)

    @staticmethod
    def get_pdf_text(pdf_path):
        """
        Retrieve the text of a PDF file using its path.

        Args:
            pdf_path (str): The path of the PDF file.

        Returns:
            A string representing the text of the PDF file.
        """
        reader = PdfReader(pdf_path)

        # get the number of pages
        num_pg = len(reader.pages)

        all_content = list()
        for i in range(num_pg):
            page = reader.pages[i]
            text = page.extract_text()
            all_content.append(text)

        all_content = " ".join(all_content)
        return all_content
