# tiny-exec-assistant-bot

## Run

Start the Telegram bot from the repo root:

```bash
make run
```

## Test

Run the smoke suite:

```bash
make test
```

Watch Python files under `src/` and `tests/` and rerun the suite after every change:

```bash
make watch
```

To run the suite automatically before each commit, point Git at the repo hook directory once:

```bash
make hooks
```

The same smoke suite also runs in GitHub Actions on every push and pull request.

## Extending the suite

Add coverage beside the affected area under `tests/smoke/` when behavior changes:

- `test_router.py` for tool-decision and fallback behavior
- `test_tools.py` for note/reminder tool contracts
- `test_bot.py` for handler orchestration and startup wiring

The tests do not modify themselves. The architecture is set up so new features fail fast when their contracts change, and the watcher/CI path keeps the suite running continuously.
