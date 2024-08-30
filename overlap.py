import cupy as cp
from tqdm import tqdm
import time

def main() -> None:
    n = 5000

    with cp.cuda.Device(0):
        x: cp.ndarray = cp.random.rand(n, n)
        y: cp.ndarray = cp.random.rand(n, n)
    with cp.cuda.Device(1):
        v = cp.random.rand(n, n)
        w = cp.empty_like(x)

    all_start = time.time()

    it = 10
    for _ in tqdm(range(it)):
        with cp.cuda.Device(0):
            x = cp.matmul(x, y)
            x = cp.matmul(x, y)
            x = cp.matmul(x, y)
            # x = cp.matmul(x, y)
        with cp.cuda.Device(1):
            cp.cuda.runtime.memcpyPeerAsync(w.data.ptr, 1, x.data.ptr, 0, x.nbytes, 1)
            cp.matmul(w, v)

    cp.cuda.runtime.streamSynchronize(0)

    all_end = time.time()
    all_lat = all_end - all_start
    print(all_lat)

if __name__ == "__main__":
    main()