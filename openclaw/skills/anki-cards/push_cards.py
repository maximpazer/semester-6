#!/usr/bin/env python3
"""Push Q/A cards into Anki via AnkiConnect.

Usage:
    python3 push_cards.py <DECK> <cards.json>

cards.json format:
    [{"front": "...", "back": "..."}, ...]
"""
import json
import sys
import urllib.request

ANKI_CONNECT_URL = "http://localhost:8765"


def anki(action, **params):
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode("utf-8")
    req = urllib.request.Request(ANKI_CONNECT_URL, data=payload)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        print(f"Fehler: konnte AnkiConnect unter {ANKI_CONNECT_URL} nicht erreichen ({exc}).", file=sys.stderr)
        sys.exit(1)

    if body.get("error") is not None:
        print(f"Fehler von AnkiConnect bei Aktion '{action}': {body['error']}", file=sys.stderr)
        sys.exit(1)

    return body.get("result")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 push_cards.py <DECK> <cards.json>", file=sys.stderr)
        sys.exit(1)

    deck, cards_path = sys.argv[1], sys.argv[2]

    try:
        with open(cards_path, "r", encoding="utf-8") as f:
            cards = json.load(f)
    except Exception as exc:
        print(f"Fehler: konnte '{cards_path}' nicht lesen ({exc}).", file=sys.stderr)
        sys.exit(1)

    anki("createDeck", deck=deck)

    notes = [
        {
            "deckName": deck,
            "modelName": "Basic",
            "fields": {"Front": card["front"], "Back": card["back"]},
            "options": {"allowDuplicate": False, "duplicateScope": "deck"},
            "tags": ["openclaw"],
        }
        for card in cards
    ]

    results = anki("addNotes", notes=notes)

    added = sum(1 for note_id in results if note_id is not None)
    skipped = sum(1 for note_id in results if note_id is None)

    print(f"OK: {added} Karten in '{deck}', {skipped} Dupes übersprungen.")


if __name__ == "__main__":
    main()
