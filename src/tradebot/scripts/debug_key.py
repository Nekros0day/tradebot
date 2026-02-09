from __future__ import annotations

from tradebot.config import Settings


def main():
    s = Settings.load()
    k = s.api_private_key
    head = k.splitlines()[0] if k else ""
    tail = k.splitlines()[-1] if k else ""
    print("First line:", head)
    print("Last line :", tail)
    print("Line count:", len(k.splitlines()))
    print("Contains CR:", "\\r" in k)
    print("Contains literal \\n:", "\\n" in k)


if __name__ == "__main__":
    main()