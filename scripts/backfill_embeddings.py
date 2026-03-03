import time

from src.embeddings import embed_text
from src.memory.store import (
    embedding_exists,
    init_db,
    iter_chat_messages,
    save_message_embedding,
)


def backfill_embeddings(batch_size: int = 100, sleep_seconds: float = 0.05) -> dict:
    init_db()

    scanned = 0
    embedded = 0
    skipped = 0
    last_seen_id = 0

    while True:
        rows = iter_chat_messages(after_id=last_seen_id, batch_size=batch_size)
        if not rows:
            break

        for row in rows:
            message_id = int(row["id"])
            chat_id = int(row["chat_id"])
            role = str(row.get("role", "")).strip()
            content = str(row.get("content", "")).strip()
            last_seen_id = message_id
            scanned += 1

            if role == "system" or not content:
                skipped += 1
            elif embedding_exists(chat_id, message_id):
                skipped += 1
            else:
                embedding = embed_text(content)
                save_message_embedding(chat_id, message_id, content, embedding)
                embedded += 1
                time.sleep(sleep_seconds)

            if scanned % 100 == 0:
                print(
                    f"backfill progress scanned={scanned} embedded={embedded} skipped={skipped} last_id={last_seen_id}"
                )

    print(
        f"backfill complete scanned={scanned} embedded={embedded} skipped={skipped} last_id={last_seen_id}"
    )
    return {
        "scanned": scanned,
        "embedded": embedded,
        "skipped": skipped,
        "last_seen_id": last_seen_id,
    }


def main() -> None:
    backfill_embeddings()


if __name__ == "__main__":
    main()
