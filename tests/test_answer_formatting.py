import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.answer_formatting import format_model_answer


class TestInlineNumberedList(unittest.TestCase):
    def test_converts_to_markdown_bullets(self):
        answer = (
            "Apple faces several risks: (1) supply chain disruptions could reduce margins, "
            "(2) cybersecurity threats are increasing, (3) regulatory changes may impact operations."
        )
        result = format_model_answer(answer)
        self.assertIn("- supply chain disruptions could reduce margins", result)
        self.assertIn("- cybersecurity threats are increasing", result)
        self.assertIn("- regulatory changes may impact operations.", result)
        self.assertNotIn("(1)", result)
        self.assertNotIn("(2)", result)
        self.assertNotIn("(3)", result)

    def test_preserves_citations_in_bullets(self):
        answer = (
            "Risks include: (1) supply chain issues [AAPL-2024-11-01-0007], "
            "(2) regulatory pressure [AAPL-2024-11-01-0008]."
        )
        result = format_model_answer(answer)
        self.assertIn("[AAPL-2024-11-01-0007]", result)
        self.assertIn("[AAPL-2024-11-01-0008]", result)

    def test_single_item_not_converted(self):
        answer = "The company reported (1) one notable risk: supply chain delays."
        result = format_model_answer(answer)
        self.assertNotIn("- ", result)


class TestTotalNetSalesExtraction(unittest.TestCase):
    def test_table_spill_becomes_scalar_sentence(self):
        question = "What were Apple's total net sales?"
        answer = "Total net sales    $416,161    $391,035    $383,285 [AAPL-2025-10-31-0046]"
        result = format_model_answer(answer, question=question)
        self.assertIn("416,161", result)
        self.assertIn("[AAPL-2025-10-31-0046]", result)
        self.assertNotIn("391,035", result)
        self.assertNotIn("383,285", result)

    def test_table_without_dollar_sign(self):
        question = "What are total net sales for AAPL?"
        answer = "Total net sales      416,161      391,035 [AAPL-2025-10-31-0046]"
        result = format_model_answer(answer, question=question)
        self.assertIn("416,161", result)
        self.assertNotIn("391,035", result)

    def test_non_sales_question_skips_extraction(self):
        question = "What risks did Apple report?"
        answer = "Total net sales    $416,161 [AAPL-2025-10-31-0046]"
        result = format_model_answer(answer, question=question)
        self.assertNotIn("reported total net sales", result)


class TestRolePrefixStripping(unittest.TestCase):
    def test_strips_answer_prefix(self):
        result = format_model_answer("Answer: Apple grew revenue significantly.")
        self.assertFalse(result.startswith("Answer:"))

    def test_strips_assistant_prefix(self):
        result = format_model_answer("Assistant: Revenue increased in FY2024.")
        self.assertFalse(result.startswith("Assistant:"))


class TestDeduplication(unittest.TestCase):
    def test_removes_duplicate_bullets(self):
        answer = "- Apple reported strong growth.\n- Apple reported strong growth.\n- Revenue increased."
        result = format_model_answer(answer)
        self.assertEqual(result.count("Apple reported strong growth."), 1)


if __name__ == "__main__":
    unittest.main()
