"""Circular block bootstrap for multivariate time-series data augmentation."""

import numpy as np
import pandas as pd


class CircularBlockBootstrap:
    """Multivariate circular block bootstrap for time-series data.

    Preserves cross-sectional dependence by bootstrapping all series
    jointly (same block indices applied to every column).  The wrap-around
    property of circular bootstrapping eliminates edge effects.

    Block length is set to ~80 % of the dataset length, which sits in the
    middle of the 70–90 % recommendation for preserving autocorrelation
    structure in financial return series.

    Args:
        block_length_ratio: Fraction of dataset length to use as block size.
            Default is 0.80.
        seed: Base random seed for reproducibility.
    """

    def __init__(self, block_length_ratio: float = 0.80, seed: int = 42) -> None:
        self.block_length_ratio = block_length_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def bootstrap(
        self, data: pd.DataFrame, target_length: int | None = None
    ) -> pd.DataFrame:
        """Generate one synthetic dataset via circular block bootstrap.

        Blocks are drawn by sampling a uniformly random start position and
        reading ``block_length`` consecutive rows with circular (modular)
        index wrap-around.  Blocks are concatenated until ``target_length``
        rows have been collected.

        Args:
            data: DataFrame where rows are time steps and columns are series.
                All columns are bootstrapped jointly.
            target_length: Number of rows in the output.  Defaults to
                ``len(data)`` if not specified.

        Returns:
            Bootstrapped DataFrame with the same columns as *data* and
            ``target_length`` rows, with a reset integer index.
        """
        T = len(data)
        if target_length is None:
            target_length = T
        block_length = max(1, int(T * self.block_length_ratio))

        arr = data.values  # shape (T, n_cols)
        result: list[np.ndarray] = []
        n_generated = 0

        while n_generated < target_length:
            start = int(self.rng.integers(0, T))
            remaining = target_length - n_generated
            take = min(block_length, remaining)

            # Circular wrap-around index
            indices = [(start + i) % T for i in range(take)]
            block = arr[indices]
            result.append(block)
            n_generated += take

        synthetic = np.vstack(result)[:target_length]
        return pd.DataFrame(synthetic, columns=data.columns)

    def regenerate(self, data: pd.DataFrame, seed_offset: int = 0) -> pd.DataFrame:
        """Regenerate a bootstrap sample with a fresh seed.

        Each call resets the internal RNG to ``self.seed + seed_offset``
        before drawing samples, ensuring reproducibility per offset value.

        Args:
            data: DataFrame to bootstrap (rows = time steps).
            seed_offset: Added to the base seed so different offsets produce
                independent realisations.

        Returns:
            Bootstrapped DataFrame with same columns as *data* and reset index.
        """
        self.rng = np.random.default_rng(self.seed + seed_offset)
        return self.bootstrap(data)
