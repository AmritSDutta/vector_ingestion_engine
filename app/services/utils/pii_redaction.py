import asyncio
import logging
from typing import List, Tuple

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from app.config.config import get_settings


class PII_Redactor:
    """
    Helps in doing redaction of given langchain based messages
    """

    def __init__(self, confidence_threshold: float = get_settings().PII_CONFIDENCE_THRESHOLD):
        self.analyzer = AnalyzerEngine(supported_languages=["en"])
        self.anonymizer = AnonymizerEngine()
        self.confidence_threshold = confidence_threshold

    async def _sanitize(self, text: str, analysis: List[RecognizerResult] | None = None) -> str:
        def _run():
            return self.anonymizer.anonymize(
                text=text,
                analyzer_results=analysis if analysis else self.analyzer.analyze(text=text, language="en"),
            ).text

        return await asyncio.to_thread(_run)

    async def _is_pii_data_detected(self, text: str) -> Tuple[List[RecognizerResult], bool]:
        def _run():
            return self.analyzer.analyze(text=text, language="en")

        analyzer_results: List[RecognizerResult] = await asyncio.to_thread(_run)
        detected: bool = any(r.score >= self.confidence_threshold for r in analyzer_results)
        return analyzer_results, detected

    async def do_pii_redaction_text(self, messages: List[str]) -> List[str]:
        """
        It first analyses whether to perform anonymization and then perform anonymization.
        It is only effects textual messages.
        """
        settings = get_settings()
        if not settings.IS_PII_REDACTION_ENABLED:
            logging.info("PII redaction disabled")
            return messages
        # logging.info("PII redaction started")
        redacted_messages: List[str] = [
            await self._do_pii_redaction_on_message_text(msg) for msg in messages
        ]
        logging.info("PII redaction completed")
        return redacted_messages

    async def _do_pii_redaction_on_message_text(self, content: str) -> str:
        if isinstance(content, str):
            analysis, detected = await self._is_pii_data_detected(content)
            if not detected:
                logging.info("PII redaction not required")
                return content

            redacted: str = await self._sanitize(content, analysis)
            return redacted

        elif isinstance(content, list):
            new_content = []
            for item in content:
                if hasattr(item, "get") and item.get("type") == "text":
                    original = item.get("text", "")
                    analysis, detected = await self._is_pii_data_detected(original)
                    if detected:
                        redacted = await self._sanitize(original, analysis)
                        item = dict(item)  # shallow copy
                        item["text"] = redacted
                        logging.info(f"user req (redacted): {redacted}")
                new_content.append(item)
            redacted_content = ''.join(new_content)
            return redacted_content

        return content
