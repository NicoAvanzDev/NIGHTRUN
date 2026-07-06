#!/usr/bin/env python3
"""Generate tokenizer reference fixtures using HF `tokenizers`.

Usage: gen_tokenizer_fixtures.py <tokenizer.json> <out.txt>
Output lines: <hex-utf8-of-text><TAB><comma-separated-ids>
"""
import sys

from tokenizers import Tokenizer

CASES = [
    "Hello, world!",
    "hello",
    " hello",
    "  leading spaces",
    "    four spaces then word",
    "trailing space ",
    "trailing spaces   ",
    "it's",
    "IT'S",
    "don't you'll we're I've he'd she'M",
    "1",
    "42",
    "123",
    "1234",
    "1234567",
    "3.14159",
    "phone: +1-800-555-0199",
    "email@test.example.com",
    "line\nbreaks",
    "line\nbreaks\n\ndouble",
    "windows\r\nline endings\r\n",
    "\n",
    "\n\n\n",
    "tab\tseparated\tvalues",
    "café naïve résumé",
    "ünïcödé",
    "你好世界",
    "🦙 llama emoji 🌆🌴",
    "fn main() { println!(\"hi\"); }",
    "x = y * 2 + z[0]",
    "The 1980s: neon & grids!!!",
    "<html><body class=\"x\">",
    "a",
    " ",
    "   word   word  ",
    "ALL CAPS SENTENCE",
    "MixedCaseWords TogetherHere",
    "punctuation... ellipsis?! yes--no",
    "«angle quotes» and “smart quotes”",
    "semi;colon:list,of,things",
    "What is the capital of France?",
    "Write a haiku about neon sunsets.",
]


def main() -> None:
    tok = Tokenizer.from_file(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        for text in CASES:
            ids = tok.encode(text, add_special_tokens=False).ids
            f.write(text.encode("utf-8").hex() + "\t" + ",".join(map(str, ids)) + "\n")
    print(f"wrote {len(CASES)} cases to {sys.argv[2]}")


if __name__ == "__main__":
    main()
