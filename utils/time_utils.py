"""Small formatting helpers."""
def format_seconds(value: float) -> str:
    minutes, seconds = divmod(max(0, int(value)), 60)
    return f"{minutes:02d}:{seconds:02d}"
