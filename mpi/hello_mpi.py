from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

# Barrier makes output easier to read by aligning process progress.
comm.Barrier()
print(f"[hello] rank={rank}/{size - 1} host={host}")
