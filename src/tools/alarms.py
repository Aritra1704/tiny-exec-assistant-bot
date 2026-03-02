from datetime import datetime, timedelta, timezone
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler

_scheduler: BackgroundScheduler | None = None


def _get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone=timezone.utc)
        _scheduler.start()
    return _scheduler


def schedule_reminder(
    send_fn: Callable[[int, str], None],
    chat_id: int,
    in_minutes: int,
    message: str,
) -> dict:
    safe_minutes = int(in_minutes)
    cleaned_message = message.strip()

    if safe_minutes <= 0:
        return {"ok": False, "error": "Reminder time must be greater than 0 minutes."}
    if not cleaned_message:
        return {"ok": False, "error": "Reminder message cannot be empty."}

    run_at = datetime.now(timezone.utc) + timedelta(minutes=safe_minutes)

    def _fire() -> None:
        send_fn(chat_id, f"Reminder: {cleaned_message}")

    job = _get_scheduler().add_job(_fire, "date", run_date=run_at)
    return {
        "ok": True,
        "job_id": job.id,
        "run_at": run_at.isoformat(timespec="seconds"),
        "in_minutes": safe_minutes,
        "message": cleaned_message,
    }
