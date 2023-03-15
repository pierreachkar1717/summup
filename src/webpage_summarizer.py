""" This module contains the WebpageSummarizer class, which can summarize a given webpage. """
from summarizer import Summarizer
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment


class WebpageSummarizer(Summarizer):
    """A class that summarizes a given webpage."""

    def __init__(self, method, url):
        self.text = self.get_url_content(url)
        self.method = method
        self.url = url
        self.chunks = self.preprocess_and_chunk_text(self.text, self.method)

    @staticmethod
    def get_url_content(url):
        """a function that takes a url and returns the content of the webpage"""

        def tag_visible(element):
            if element.parent.name in [
                "style",
                "script",
                "head",
                "title",
                "meta",
                "[document]",
            ]:
                return False
            if isinstance(element, Comment):
                return False
            return True

        html = requests.get(url)
        soup = BeautifulSoup(html.content, "html.parser")
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        return " ".join(t.strip() for t in visible_texts)


test = WebpageSummarizer("transformers", "https://www.tagesschau.de/")
sum = test.summarize()
print(sum)
