from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

scheduler = BackgroundScheduler()
scheduler.start()

def schedule_reminder(send_fn, chat_id: int, in_minutes: int, message: str):
    run_at = datetime.now() + timedelta(minutes=in_minutes)

    def _fire():
        send_fn(chat_id, f"⏰ Reminder: {message}")

    job = scheduler.add_job(_fire, "date", run_date=run_at)
    return {
        "ok": True,
        "job_id": job.id,
        "run_at": run_at.isoformat(timespec="seconds"),
        "message": message,
    }