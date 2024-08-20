"""
Claude API call tests
"""

import os

import anthropic

client = anthropic.Anthropic()

if __name__ == "__main__":

    res = anthropic.Anthropic(default_headers={"anthropic-version": "2023-06-01"}).messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, world"}],
    )

    for idx, textblock in enumerate(res.content):
        print(idx, textblock.text)

    with client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-3-5-sonnet-20240620",
    ) as stream:
        for idx, text in enumerate(stream.text_stream):
            print(idx, text, end="", flush=True)
            # print(idx, text, flush=True)
