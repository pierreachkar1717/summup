""" 
A module that summarizes a given YouTube video using a specified summarization method.
"""
from summarizer import Summarizer
from youtube_transcript_api import YouTubeTranscriptApi


class VideoSummarizer(Summarizer):
    """
    A class that summarizes a given YouTube video using a specified summarization method.

    Attributes:
        url (str): The URL of the YouTube video.
        text (str): The transcript of the YouTube video.
        method (str): The summarization method.
        chunks (list of str): A list of preprocessed text chunks, where each chunk
        contains only full sentences.
    """

    def __init__(self, url, method):
        self.url = url
        self.text = self.get_video_transcript(url)
        self.method = method
        self.chunks = self.preprocess_and_chunk_text(self.text, self.method)

    @staticmethod
    def get_video_transcript(video_url):
        """
        Retrieve the transcript of a YouTube video using its URL, and its text.

        Args:
            video_url (str): The URL of the YouTube video.

        Returns:
            A dictionary with the following keys:
            - 'transcript': a list of dictionaries, where each dictionary represents a caption.
                Each dictionary has the following keys: 'text', 'start', and 'duration'.
                If the transcript cannot be retrieved, this key has a value of None.
            - 'transcript_text': a string representing the concatenation of all captions' text.
                If the transcript cannot be retrieved, this key has a value of None.
        """
        video_id = video_url.split("v=")[1]
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            captions = [caption["text"] for caption in transcript]
            transcript_text = " ".join(captions)
        except YouTubeTranscriptApi.exceptions.TranscriptNotFoundError:
            transcript = None
            transcript_text = None
        return transcript_text


# TODO: Add Whisper to transcript youtube videos that don't have transcripts

test = VideoSummarizer("https://www.youtube.com/watch?v=OU72wcPyUfM", "transformers")
print(test.summarize())
