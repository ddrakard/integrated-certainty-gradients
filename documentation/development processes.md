# Development processes

## Testing

Run the tests by executing `python3 -m unittest` from inside the `src` directory.

## Static analysis

### Check style / formatting

Run `pycodestyle .` from directory `src`.
Pycodestyle requires installation through pip.

### Check code quality

Run `pyflakes .` from directory `src`.
Pyflakes requires installation through pip.

Run `python3 run_pylint.py` from directory `src`.

Run `mypy --ignore-missing-imports --disallow-incomplete-defs .` from directory `src`.
Mypy requires installation through pip.
