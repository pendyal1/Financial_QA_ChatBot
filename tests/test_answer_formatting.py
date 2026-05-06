from __future__ import annotations

import unittest

from finrag.answer_formatting import format_model_answer


class AnswerFormattingTest(unittest.TestCase):
    def test_formats_inline_numbered_list_as_markdown_bullets(self) -> None:
        answer = (
            "Microsoft describes these risks: (1) evolving cyberthreats "
            "(2) AI-enabled attacks (3) AI-enabled attacks (4) increased regulation"
        )

        formatted = format_model_answer(answer)

        self.assertEqual(
            formatted,
            "\n".join(
                [
                    "Microsoft describes these risks:",
                    "",
                    "- evolving cyberthreats.",
                    "- AI-enabled attacks.",
                    "- increased regulation.",
                ]
            ),
        )

    def test_repairs_total_net_sales_context_spill(self) -> None:
        answer = (
            "| 2025 Form 10-K | Products and Services Performance "
            "The following table shows net sales by category (dollars in millions): "
            "iPhone 209,586 Mac 33,708 Total net sales 416,161 391,035 "
            "[AAPL-2025-10-31-0046]"
        )

        formatted = format_model_answer(
            answer,
            question="What was AAPL total net sales amount in millions for fiscal year 2025?",
        )

        self.assertEqual(
            formatted,
            "Apple reported total net sales of $416,161 million for fiscal year 2025. "
            "[AAPL-2025-10-31-0046]",
        )


if __name__ == "__main__":
    unittest.main()
