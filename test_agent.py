import unittest
import re
from agent import classify_intent_local

class TestAgentLogic(unittest.TestCase):
    def test_classify_intent_general_qa(self):
        # Test basic question intent routing
        intent = classify_intent_local("What is quantum computing?", [], "What is quantum computing?")
        self.assertEqual(intent, "TASK: answer")

    def test_classify_intent_summarize(self):
        # Test summarization keyword trigger
        intent = classify_intent_local("summarize this text please", [], "summarize this text please")
        self.assertEqual(intent, "TASK: summarize")

    def test_classify_intent_code(self):
        # Test code explanation trigger
        intent = classify_intent_local("explain this Python code", [], "explain this Python code")
        self.assertEqual(intent, "TASK: explain_code")

    def test_classify_intent_compare(self):
        # Test comparison trigger
        intent = classify_intent_local("compare these files", [], "compare these files")
        self.assertEqual(intent, "TASK: compare")

    def test_classify_intent_empty_query_with_files(self):
        # Test the auto-chain summary rule for uploads with empty queries
        mock_extracted = [{"source": "lecture.mp3", "text": "transcription content"}]
        intent = classify_intent_local("", mock_extracted, "[From lecture.mp3]:\ntranscription content")
        self.assertEqual(intent, "TASK: summarize")

    def test_classify_intent_empty_query_no_files(self):
        # Test that empty query without files requests clarification
        intent = classify_intent_local("", [], "")
        self.assertTrue(intent.startswith("CLARIFY:"))

    def test_youtube_url_extraction(self):
        # Verify YouTube URL parsing pattern
        url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        url2 = "https://youtu.be/dQw4w9WgXcQ"
        
        patterns = [
            r'youtube\.com/watch\?v=([\w\-]+)',
            r'youtu\.be/([\w\-]+)'
        ]
        
        id1 = None
        for p in patterns:
            match = re.search(p, url1)
            if match:
                id1 = match.group(1)
                break
        self.assertEqual(id1, "dQw4w9WgXcQ")

        id2 = None
        for p in patterns:
            match = re.search(p, url2)
            if match:
                id2 = match.group(1)
                break
        self.assertEqual(id2, "dQw4w9WgXcQ")

if __name__ == "__main__":
    unittest.main()
