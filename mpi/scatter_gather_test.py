from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 prepares balanced chunks so scatter works even when length is not size-multiple.
if rank == 0:
    total_items = 23
    data = np.arange(total_items, dtype=np.int32)
    chunks = np.array_split(data, size)
    print(f"[scatter] total_items={total_items}, ranks={size}")
else:
    chunks = None

local_chunk = comm.scatter(chunks, root=0)

# Local compute placeholder for image-block processing.
local_result = local_chunk * 2

gathered = comm.gather(local_result, root=0)

if rank == 0:
    final = np.concatenate(gathered)
    expected = np.arange(23, dtype=np.int32) * 2
    ok = np.array_equal(final, expected)
    print(f"[gather] collected={final.size}, validation={'PASS' if ok else 'FAIL'}")
