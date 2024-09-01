import cupy as cp
from tqdm import tqdm
import time

def main() -> None:
    n = 5000

    with cp.cuda.Device(0):
        stream_gpu0 = cp.cuda.Stream()
    with cp.cuda.Device(1):
        stream_gpu1 = cp.cuda.Stream()

    with cp.cuda.Device(0):
        x: cp.ndarray = cp.random.rand(n, n)
        y: cp.ndarray = cp.random.rand(n, n)
        cp.cuda.Device().synchronize()
    with cp.cuda.Device(1):
        v = cp.random.rand(n, n)
        w = cp.empty_like(x)
        cp.cuda.Device().synchronize()

    all_start = time.time()

    # cp.cuda.runtime.profilerStart()

    """
    0: _ _____ _ _____ _ _____
    1:         _       _      _

    vs.

    0: _ _____ _____ _____
         _     _
    1:        _     _      _

    vs.

    0: _ _____
         _ _____
           _ _____
    1:        _ _ _
    """

    it = 10
    for _ in tqdm(range(it)):
        with cp.cuda.Device(0):
            with stream_gpu0:
                x = cp.matmul(x, y)
                x = cp.matmul(x, y)
                x = cp.matmul(x, y)
                x = cp.matmul(x, y)
                # evt = cp.cuda.runtime.eventCreate()
                cp.cuda.runtime.memcpyPeer(w.data.ptr, 1, x.data.ptr, 0, x.nbytes)
                cp.cuda.runtime.streamSynchronize(0)
                # cp.cuda.runtime.eventSynchronize(evt)
        with cp.cuda.Device(1):
            # w = x.copy()
            with stream_gpu1:
                cp.matmul(w, v)

    cp.cuda.runtime.streamSynchronize(0)
    # cp.cuda.runtime.profilerStop()

    all_end = time.time()
    all_lat = all_end - all_start
    print(all_lat)

if __name__ == "__main__":
    main()
