import cupy as cp
from enum import Enum
from tqdm import tqdm
import time
from typing import *

heavy_add = cp.RawKernel(
    r"""
extern "C" __global__
void heavy_add(float* A, float* B, float* C, int N, int cycles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        for (int i = 0; i < cycles; i++) {
            float sinA = sinf(A[tid]);
            float cosA = cosf(A[tid]);
            float idA = sqrtf(sinA * sinA + cosA * cosA) * A[tid];
            float sinB = sinf(B[tid]);
            float cosB = cosf(B[tid]);
            float idB = sqrtf(sinB * sinB + cosB * cosB) * B[tid];
            float added = idA + idB;
            float sinAdded = sinf(added);
            float cosAdded = cosf(added);
            float idAdded = sqrtf(sinAdded * sinAdded + cosAdded * cosAdded) * added;
            C[tid] = idAdded;
        }
    }
}
""",
    "heavy_add",
)


class IssueOrder(Enum):
    SEQ = r"{comp_0^0, comm_0^0, comp_0^1} {comp_1^0, comm_1^0, comp_1^1}"
    GROUP = r"{comp_0^0, comp_1^0} {comm_0^0, comm_1^0} {comp_0^1, comp_1^1}"


def experiment(
    overlap: bool, N: int, iters: int, cycles: int, order: IssueOrder
) -> float:
    BLOCK: int = 256
    GRID: int = (N + BLOCK - 1) // BLOCK
    SIZE: int = N * 4

    # create vectors
    with cp.cuda.Device(0):
        a: cp.ndarray = cp.full(N, 1.0, dtype=float)
        b: cp.ndarray = cp.full(N, 2.0, dtype=float)
        # initialize c, x, f, y to some value for easier debugging if validation fails
        c: cp.ndarray = cp.full(N, -100.0, dtype=float)
        x: cp.ndarray = cp.full(N, -10.0, dtype=float)
    with cp.cuda.Device(1):
        d: cp.ndarray = cp.full(N, 3.0, dtype=float)
        e: cp.ndarray = cp.full(N, 0.0, dtype=float)
        f: cp.ndarray = cp.full(N, -50.0, dtype=float)
        y: cp.ndarray = cp.full(N, -5.0, dtype=float)

    # create events and streams
    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    with cp.cuda.Device(0):
        comp_str_0 = cp.cuda.Stream()
        comm_str = cp.cuda.Stream()
        comp_c_evt = cp.cuda.Event()
        comm_c_evt = cp.cuda.Event()
        comp_x_evt = cp.cuda.Event()
        comm_x_evt = cp.cuda.Event()
    with cp.cuda.Device(1):
        comp_str_1 = cp.cuda.Stream()
        comp_f_evt = cp.cuda.Event()
        comp_y_evt = cp.cuda.Event()
        sync_evt = cp.cuda.Event()

    # timing
    start_evt.record()
    start_time: float = time.time()

    # for _ in tqdm(range(iters // 2)):
    for _ in range(iters // 2):
        match order:
            case IssueOrder.SEQ:
                with cp.cuda.Device(0):
                    comp_str_0.wait_event(comm_c_evt)
                    with comp_str_0:
                        heavy_add((GRID,), (BLOCK,), (a, b, c, N, cycles))
                    if overlap:
                        comp_c_evt.record(comp_str_0)
                        comm_str.wait_event(comp_c_evt)
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, c.data.ptr, 0, SIZE, comm_str.ptr
                        )
                        comm_c_evt.record(comm_str)
                    else:
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, c.data.ptr, 0, SIZE, comp_str_0.ptr
                        )
                        comm_c_evt.record(comp_str_0)

                with cp.cuda.Device(1):
                    comp_str_1.wait_event(comm_c_evt)
                    comp_str_1.wait_event(comp_f_evt)
                    with comp_str_1:
                        heavy_add((GRID,), (BLOCK,), (d, e, f, N, cycles))
                    comp_f_evt.record(comp_str_1)

                with cp.cuda.Device(0):
                    comp_str_0.wait_event(comm_x_evt)
                    with comp_str_0:
                        heavy_add((GRID,), (BLOCK,), (a, b, x, N, cycles))
                    if overlap:
                        comp_x_evt.record(comp_str_0)
                        comm_str.wait_event(comp_x_evt)
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, x.data.ptr, 0, SIZE, comm_str.ptr
                        )
                        comm_x_evt.record(comm_str)
                    else:
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, x.data.ptr, 0, SIZE, comp_str_0.ptr
                        )
                        comm_x_evt.record(comp_str_0)

                with cp.cuda.Device(1):
                    comp_str_1.wait_event(comm_x_evt)
                    comp_str_1.wait_event(comp_y_evt)
                    with comp_str_1:
                        heavy_add((GRID,), (BLOCK,), (d, e, y, N, cycles))
                    comp_y_evt.record(comp_str_1)

            case IssueOrder.GROUP:
                with cp.cuda.Device(0):
                    comp_str_0.wait_event(comm_c_evt)
                    with comp_str_0:
                        # cp.add(a, b, c)
                        heavy_add((GRID,), (BLOCK,), (a, b, c, N, cycles))
                    if overlap:
                        comp_c_evt.record(comp_str_0)
                    else:
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, c.data.ptr, 0, SIZE, comp_str_0.ptr
                        )
                        comm_c_evt.record(comp_str_0)

                    comp_str_0.wait_event(comm_x_evt)
                    with comp_str_0:
                        heavy_add((GRID,), (BLOCK,), (a, b, x, N, cycles))
                    if overlap:
                        comp_x_evt.record(comp_str_0)
                    else:
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, x.data.ptr, 0, SIZE, comp_str_0.ptr
                        )
                        comm_x_evt.record(comp_str_0)

                    if overlap:
                        comm_str.wait_event(comp_c_evt)
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, c.data.ptr, 0, SIZE, comm_str.ptr
                        )
                        comm_c_evt.record(comm_str)

                        comm_str.wait_event(comp_x_evt)
                        cp.cuda.runtime.memcpyPeerAsync(
                            e.data.ptr, 1, x.data.ptr, 0, SIZE, comm_str.ptr
                        )
                        comm_x_evt.record(comm_str)

                with cp.cuda.Device(1):
                    comp_str_1.wait_event(comm_c_evt)
                    comp_str_1.wait_event(comp_f_evt)
                    with comp_str_1:
                        heavy_add((GRID,), (BLOCK,), (d, e, f, N, cycles))
                    comp_f_evt.record(comp_str_1)

                    comp_str_1.wait_event(comm_x_evt)
                    comp_str_1.wait_event(comp_y_evt)
                    with comp_str_1:
                        heavy_add((GRID,), (BLOCK,), (d, e, y, N, cycles))
                    comp_y_evt.record(comp_str_1)

    if iters % 2 == 1:
        with cp.cuda.Device(0):
            comp_str_0.wait_event(comm_c_evt)
            with comp_str_0:
                heavy_add((GRID,), (BLOCK,), (a, b, c, N, cycles))
            if overlap:
                comp_c_evt.record(comp_str_0)
                comm_str.wait_event(comp_c_evt)
                cp.cuda.runtime.memcpyPeerAsync(
                    e.data.ptr, 1, c.data.ptr, 0, SIZE, comm_str.ptr
                )
                comm_c_evt.record(comm_str)
            else:
                cp.cuda.runtime.memcpyPeerAsync(
                    e.data.ptr, 1, c.data.ptr, 0, SIZE, comp_str_0.ptr
                )
                comm_c_evt.record(comp_str_0)

        with cp.cuda.Device(1):
            comp_str_1.wait_event(comm_c_evt)
            comp_str_1.wait_event(comp_f_evt)
            with comp_str_1:
                heavy_add((GRID,), (BLOCK,), (d, e, f, N, cycles))
            comp_f_evt.record(comp_str_1)

    with cp.cuda.Device(1):
        sync_evt.record(comp_str_1)
        sync_evt.synchronize()

    end_evt.record()

    end_time: float = time.time()
    time_elapse: float = end_time - start_time
    print(f"python timer: {time_elapse}")

    cuda_elapse = cp.cuda.get_elapsed_time(start_evt, end_evt)
    print(f"cuda timer: {cuda_elapse}")

    return cuda_elapse


def run_cmp_expr(N: int, iters: int, cycles: int, order: IssueOrder) -> None:
    print_expr_settings(N, iters, cycles, order)
    print("Running synchronous version...")
    sync: float = experiment(False, N, iters, cycles, order)
    print("Running overlapping version...")
    overlap: float = experiment(True, N, iters, cycles, order)
    speedup: float = (sync - overlap) / sync
    print(f"speedup: {speedup}")


def print_expr_settings(N: int, iters: int, cycles: int, order: IssueOrder) -> None:
    print("==============================")
    print(f"N = {N}, #it = {iters}, cycles = {cycles}, order = {order}")


def main() -> None:
    BEST_CYCLES: int = 32
    BEST_ITERS: int = 100
    BEST_N: int = 1_000_000
    # BEST_ORDER: IssueOrder = IssueOrder.SEQ
    BEST_ORDER: IssueOrder = IssueOrder.GROUP

    N_exp_trials: int = 8
    N_exp_base: int = 1
    N_exp_step: int = 10

    iters_exp_trials: int = 10
    iters_exp_base: int = 1
    iters_exp_step: int = 2

    cycles_exp_trials: int = 8
    cycles_exp_base: int = 1
    cycles_exp_step: int = 2

    cycles_inc_trials: int = 10
    cycles_inc_base: int = 10
    cycles_inc_step: int = 10

    print("Changing input sizes:")
    print()
    # for _ in tqdm(range(N_exp_trials)):
    for _ in range(N_exp_trials):
        run_cmp_expr(N_exp_base, BEST_ITERS, BEST_CYCLES, BEST_ORDER)
        N_exp_base *= N_exp_step
    print()

    print("Changing #iterations:")
    print()
    # for _ in tqdm(range(iters_exp_trials)):
    for _ in range(iters_exp_trials):
        run_cmp_expr(BEST_N, iters_exp_base, BEST_CYCLES, BEST_ORDER)
        iters_exp_base *= iters_exp_step
    print()

    print("Changing issue order:")
    # for order in tqdm(IssueOrder):
    for order in IssueOrder:
        print()
        print(order.value)
        run_cmp_expr(BEST_N, BEST_ITERS, BEST_CYCLES, order)
    print()

    print("Changing computation intensity (exponential):")
    print()
    # for _ in tqdm(range(cycles_exp_trials)):
    for _ in range(cycles_exp_trials):
        run_cmp_expr(BEST_N, BEST_ITERS, cycles_exp_base, BEST_ORDER)
        cycles_exp_base *= cycles_exp_step
    print()

    print("Changing computation intensity (incremental):")
    print()
    # for _ in tqdm(range(cycles_inc_trials)):
    for _ in range(cycles_inc_trials):
        run_cmp_expr(BEST_N, BEST_ITERS, cycles_inc_base, BEST_ORDER)
        cycles_inc_base += cycles_inc_step
    print()


if __name__ == "__main__":
    main()
