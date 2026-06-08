"""
Batch Sampling Utilities

In ML training loops, data is rarely processed all at once. Instead it is
split into batches and — when training neural networks — reshuffled each
epoch so that the model never sees the same ordering twice.

Below you will implement the building blocks of a lightweight data-sampling
pipeline. Together, the four components mirror the interface you will later
encounter in PyTorch's DataLoader.

NOTE: The class `EpochSampler` is an iterable but not an iterator. More on
this distinction can be found here:

    https://www.geeksforgeeks.org/python/python-difference-iterable-iterator/


DO NOT MODIFY THE FUNCTION SIGNATURES.
"""

import random


# ---------------------------------------------------------------------------
# 1. chunks
# ---------------------------------------------------------------------------


def chunks(seq, size: int):
    """
    A generator that yields successive non-overlapping chunks of `size`
    items from seq. The final chunk may be shorter than `size` if the
    sequence length is not evenly divisible.

    Arguments:
        seq  -- any sequence (list, tuple, str, range, …)
        size -- a positive integer chunk size

    Yields:
        list -- successive chunks

    Raises:
        ValueError -- if size < 1

    Example:
        >>> list(chunks([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]

    (_Hint: Use `yield` to make this a generator function._)
    """

    # ------ SOLUTION GOES HERE!  ------
    if size < 1:
        raise ValueError("size must be >= 1")

    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])

# ---------------------------------------------------------------------------
# 2. take
# ---------------------------------------------------------------------------


def take(iterable, n: int) -> list:
    """
    Consumes up to n items from iterable and returns them as a list.
    Stops early — without raising an error — if the iterable is exhausted
    before n items have been collected.

    You MUST use iter() and next() explicitly in your implementation.
    Do not use slicing, islice, or list comprehensions.
    (_Note: `next()` raises `StopIteration` when the iterator is exhausted._)

    Arguments:
        iterable -- any iterable object
        n        -- a non-negative integer

    Returns:
        list -- the first n items (or fewer) from iterable

    Raises:
        ValueError -- if n < 0

    Example:
        >>> take(range(3), 5)   # iterable runs out first
        [0, 1, 2]
        >>> take(range(10), 4)
        [0, 1, 2, 3]
    """

    # ------ SOLUTION GOES HERE!  ------
    if n < 0:
        raise ValueError("n must be >= 0")

    it = iter(iterable)
    out = []

    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break

    return out

# ---------------------------------------------------------------------------
# 3. first_satisfying
# ---------------------------------------------------------------------------


def first_satisfying(iterable, predicate) -> tuple:
    """
    Searches iterable for the first element satisfying predicate and returns
    it together with its zero-based position in the iteration order.

    You MUST use a for-else construct: if the loop completes without finding
    a match, the else branch should run and the function should return
    (None, -1).

    Arguments:
        iterable  -- any iterable
        predicate -- a callable(x) -> bool

    Returns:
        tuple(element, index) -- the first matching element and its index
        tuple(None, -1)       -- if no element satisfies predicate

    Raises:
        TypeError -- if predicate is not callable

    Example:
        >>> first_satisfying([4, 7, 2, 9], lambda x: x > 6)
        (7, 1)
        >>> first_satisfying([1, 2, 3], lambda x: x > 10)
        (None, -1)
    """

    # ------ SOLUTION GOES HERE!  ------
    if not callable(predicate):
        raise TypeError("predicate must be callable")

    for idx, element in enumerate(iterable):
        if predicate(element):
            return (element, idx)
    else:
        return (None, -1)

# ---------------------------------------------------------------------------
# 4. EpochSampler
# ---------------------------------------------------------------------------


class EpochSampler:
    """
    Iterable (not iterator) yielding shuffled full-size index batches for
    one epoch. Partial batches are dropped (drop_last).

    With seed=None each iteration produces a fresh shuffle; a fixed seed
    makes every iteration identical.

    Information on random number generators in Python can be found here:

        https://docs.python.org/3/library/random.html#alternative-generator

    Arguments:
        n_samples  -- int >= 1; total dataset size
        batch_size -- 1 <= int <= n_samples; indices per batch
        seed       -- optional int; fixes the shuffle for reproducibility

    Yields:
        list[int] -- one full batch of shuffled indices

    Raises (in __init__):
        ValueError -- if n_samples < 1 or batch_size out of range

    (_Hints: (1) Instantiate `random.Random(self.seed)` at the top of
    `__iter__` — not `__init__` — so each call is independently seeded.
    (2) Have `__iter__` return a generator object rather than containing
    `yield` directly; a bare `yield` turns `__iter__` into a generator
    method, making the sampler an iterator exhausted after one pass._)
    """

    # ------ SOLUTION GOES HERE!  ------
    def __init__(self, n_samples: int, batch_size: int, seed=None):
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if batch_size < 1 or batch_size > n_samples:
            raise ValueError("batch_size must be between 1 and n_samples")

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        # Crear un RNG independiente por iteración
        rng = random.Random(self.seed)

        def generator():
            indices = list(range(self.n_samples))
            rng.shuffle(indices)

            # drop_last: solo batches completos
            for i in range(0, self.n_samples - self.batch_size + 1, self.batch_size):
                yield indices[i:i + self.batch_size]

        return generator()

if __name__ == "__main__":
    # Synthetic dataset: 20 samples, each with a scalar "feature" value.
    # Values decrease with index so loss naturally improves as the sampler
    # draws more high-index samples in later shuffles.
    random.seed(0)
    dataset = [1.0 - (i / 20) + random.uniform(-0.05, 0.05) for i in range(20)]

    N_EPOCHS = 2
    BATCH_SIZE = 4
    LOG_INTERVAL = 2  # summarise this many batches at a time
    LOSS_TARGET = 0.55  # threshold we want to cross

    sampler = EpochSampler(n_samples=len(dataset), batch_size=BATCH_SIZE)

    for epoch in range(N_EPOCHS):
        print(f"\n{'=' * 42}")
        print(f"Epoch {epoch + 1}")

        # Materialise this epoch's batches so we can inspect and iterate them.
        batches = list(sampler)

        # --- take: preview the first two batches before training begins -------
        preview = take(batches, 2)
        print(f"  Preview — first {len(preview)} batches of indices: {preview}")

        # --- simulate a mean loss for every batch ----------------------------
        # --- (loss is a metric calculated for every data point when training
        # --- a neural network, but you'll learn more about this soon) --------
        batch_losses = [sum(dataset[i] for i in batch) / len(batch) for batch in batches]

        # --- chunks: log average loss every LOG_INTERVAL batches -------------
        """testt=(chunks(list(enumerate(batch_losses)), LOG_INTERVAL))
        for t in testt:
            print(t)"""
        print(f"  Training — {len(batches)} batches, logging every {LOG_INTERVAL}:")
        for group in chunks(list(enumerate(batch_losses)), LOG_INTERVAL):
            avg = sum(loss for _, loss in group) / len(group)
            lo, hi = group[0][0], group[-1][0]
            print(f"    batches {lo}–{hi}  avg loss: {avg:.4f}")

        # --- first_satisfying: find when loss first drops below target --------
        value, pos = first_satisfying(batch_losses, lambda loss: loss < LOSS_TARGET)
        if value is not None:
            print(f"  Loss first dropped below {LOSS_TARGET} at batch {pos} (loss={value:.4f})")
        else:
            print(f"  Loss never dropped below {LOSS_TARGET} this epoch")
