from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import os
import sys


class TqdmPool(ProcessPoolExecutor):
    def __init__(
        self,
        max_workers=multiprocessing.cpu_count() // 2,
        mp_context=multiprocessing.get_context("spawn"),
        **tqdm_kwargs,
    ):
        super().__init__(max_workers=max_workers, mp_context=mp_context)
        self.pbar = tqdm(total=1, **tqdm_kwargs)
        self.total = 0 

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)

        self.total += 1
        processed = self.pbar.n 
        self.pbar.reset(self.total) 
        self.pbar.update(processed) 
        self.pbar.refresh()  

        future.add_done_callback(self._worker_is_done)
        return future

    def _worker_is_done(self, future):
        self.pbar.update(n=1)
        self.pbar.refresh()  


def silence_output():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
