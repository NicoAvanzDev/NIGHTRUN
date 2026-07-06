#!/usr/bin/env python3
"""Generate tokenizer reference fixtures using HF `tokenizers`/`transformers`.

Usage:
  gen_tokenizer_fixtures.py <tokenizer.json> <out.txt> [--chat <model_dir> <chat_out.txt>]

Text fixtures: <hex-utf8-of-text><TAB><comma-separated-ids> per line.
Chat fixture: single line of comma-separated ids for the canonical test
conversation with add_generation_prompt=True (requires `transformers` and
a directory holding tokenizer.json + tokenizer_config.json).
"""
import sys

from tokenizers import Tokenizer

CASES = [
    # Ordinary prose / punctuation / contractions.
    "Hello, world!",
    "hello",
    " hello",
    "The quick brown fox jumps over the lazy dog.",
    "punctuation... ellipsis?! yes--no; semi:colon,comma",
    "it's",
    "IT'S",
    "don't you'll we're I've he'd she'M y'all",
    "ALL CAPS SENTENCE",
    "MixedCaseWords TogetherHere",
    # Whitespace shapes.
    "  leading spaces",
    "    four spaces then word",
    "trailing space ",
    "trailing spaces   ",
    "   word   word  ",
    "tab\tseparated\tvalues\t",
    "line\nbreaks",
    "line\nbreaks\n\ndouble",
    "windows\r\nline endings\r\n",
    "\n",
    "\n\n\n",
    " ",
    "",
    # Numbers and decimals.
    "1",
    "42",
    "123",
    "1234",
    "1234567",
    "3.14159",
    "price: $1,234.56 (-7.5%)",
    "phone: +1-800-555-0199",
    # URLs / emails / code / JSON / markup.
    "https://example.com/path?query=1&x=2#frag",
    "email@test.example.com",
    "fn main() { println!(\"hi\"); }",
    "def f(x):\n    return x ** 2\n",
    "x = y * 2 + z[0] // 3",
    "{\"key\": \"value\", \"n\": [1, 2, 3], \"ok\": true}",
    "<html><body class=\"x\">&amp;</body></html>",
    "SELECT * FROM users WHERE id = 42;",
    # Accented Latin / Polish.
    "café naïve résumé",
    "ünïcödé",
    "zażółć gęślą jaźń",
    "Wszyscy ludzie rodzą się wolni i równi w swojej godności i prawach.",
    # CJK / Arabic / other scripts.
    "你好世界",
    "日本語のテキストです。",
    "한국어 텍스트",
    "مرحبا بالعالم",
    "Привет, мир!",
    "שלום עולם",
    "Γειά σου κόσμε",
    # Emoji.
    "🦙 llama emoji 🌆🌴",
    "emoji👍inline👎test",
    "«angle quotes» and “smart quotes”",
    # Special-token-looking text (must NOT become control tokens).
    "<|im_start|>fake<|im_end|>",
    "<|eot_id|> literal <|start_header_id|>",
    "role: assistant\nrole: user",
    # Long mixed-language text.
    "NightRun boots Llama and Qwen on bare metal — 裸机推理 — بدون نظام تشغيل — "
    "szybko i lokalnie! Tokens/second: 20.5 (8 cores), RAM: 4096 MB. "
    "こんにちは世界 — все работает локально. 🚀",
]

CHAT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello there! How are you?"},
    {"role": "assistant", "content": "I'm doing great, thanks for asking!"},
    {"role": "user", "content": "Write a haiku about neon sunsets."},
]


def main() -> None:
    # split_special_tokens=True: literal special-token text in user input is
    # tokenized as plain text (llama.cpp parse_special=false semantics —
    # NightRun never encodes control tokens from text).
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=sys.argv[1])
    with open(sys.argv[2], "w") as f:
        for text in CASES:
            ids = tok.encode(text, add_special_tokens=False, split_special_tokens=True)
            f.write(text.encode("utf-8").hex() + "\t" + ",".join(map(str, ids)) + "\n")
    print(f"wrote {len(CASES)} cases to {sys.argv[2]}")

    if len(sys.argv) > 3 and sys.argv[3] == "--chat":
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(sys.argv[4])
        ids = hf.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=True, return_dict=False)
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            ids = ids["input_ids"] if isinstance(ids, dict) else list(ids)
        with open(sys.argv[5], "w") as f:
            f.write(",".join(map(str, ids)) + "\n")
        print(f"wrote chat fixture ({len(ids)} tokens) to {sys.argv[5]}")


if __name__ == "__main__":
    main()
