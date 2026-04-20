from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PrivacyShield:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact_pii(self, text: str) -> str:
        """
        Scans clinical text for PII (names, emails, phones, SSNs, locations) and
        redacts them using standard entity replacements like [PERSON] or [LOCATION].
        """
        # Analyze the text
        results = self.analyzer.analyze(text=text, language="en")
        
        # Anonymize using the analyzer results
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )
        return anonymized_result.text

# Singleton implementation
privacy_shield = PrivacyShield()
