from youtube_transcript_api import YouTubeTranscriptApi
import re

def fetch_youtube_transcript(url: str) -> str:
    try:
        # Extract video ID from URL
        patterns = [
            r'youtube\.com/watch\?v=([\w\-]+)',
            r'youtu\.be/([\w\-]+)'
        ]
        video_id = None
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break

        if not video_id:
            return "Could not extract video ID from URL."

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript_list])
        return full_text.strip()

    except Exception as e:
        return f"YouTube transcript unavailable: {str(e)}"