"""
task_select.py — shared parser for selecting tasks by number / range / list.

Lets every tool accept the SAME selection syntax instead of only --start/--tasks.

Accepted spec strings (1-based task numbers, matching tmdb_001 = 1):
    "10"          -> just task 10
    "20-25"       -> tasks 20 through 25 inclusive
    "1,5,9"       -> tasks 1, 5, and 9
    "1-3,10,20-22"-> mix of ranges and singles
    "all" / ""    -> everything (None)

Helpers:
    parse_task_spec(spec)          -> sorted list of ints, or None for "all"
    task_id_to_num("tmdb_014")     -> 14
    num_to_task_id(14)             -> "tmdb_014"
    spec_matches(task_id, nums)    -> True if this task is selected
"""
import re


def parse_task_spec(spec):
    """Return a sorted list of 1-based task numbers, or None for 'all'/empty."""
    if spec is None:
        return None
    spec = str(spec).strip().lower()
    if spec in ("", "all"):
        return None
    nums = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            nums.update(range(a, b + 1))
        elif part.isdigit():
            nums.add(int(part))
        else:
            raise ValueError(
                f"Bad task spec part: '{part}'. Use e.g. 10, 20-25, or 1,5,9.")
    return sorted(nums)


def task_id_to_num(task_id):
    """tmdb_014 -> 14 ; returns None if no trailing number."""
    m = re.search(r"(\d+)\s*$", str(task_id))
    return int(m.group(1)) if m else None


def num_to_task_id(n, prefix="tmdb"):
    return f"{prefix}_{n:03d}"


def spec_matches(task_id, nums):
    """True if task_id's number is in nums (None nums = match all)."""
    if nums is None:
        return True
    n = task_id_to_num(task_id)
    return n in nums
