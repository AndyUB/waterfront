import cupy as cp
from tqdm import tqdm
import time

def main() -> None:
    n = 5000
    it = 10

    streams = []
    with cp.cuda.Device(0):
        for _ in range(it):
            streams.append(cp.cuda.Stream())
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

    for i in tqdm(range(it)):
        with cp.cuda.Device(0):
            with streams[i] as stream:
                x = cp.matmul(x, y)
                x = cp.matmul(x, y)
                x = cp.matmul(x, y)
                x = cp.matmul(x, y)
                cp.cuda.runtime.memcpyPeer(w.data.ptr, 1, x.data.ptr, 0, x.nbytes)
                stream.synchronize()
        with cp.cuda.Device(1):
            with stream_gpu1:
                cp.matmul(w, v)

    cp.cuda.runtime.streamSynchronize(0)
    # cp.cuda.runtime.profilerStop()

    all_end = time.time()
    all_lat = all_end - all_start
    print(all_lat)

if __name__ == "__main__":
    main()
