Intelligent Edge Distributed Image Processing System
A Raspberry Pi Cluster for Parallel Image Processing
Spring 2025
Abstract
This proposal details the design, theoretical foundation, and implementation roadmap for an
Intelligent Edge Supercomputer—a 6-node Raspberry Pi cluster (1 Master, 5 Workers) dedicated
to parallel image processing. The system empirically investigates the Scale-Up (GPU) vs. ScaleOut (Cluster) paradigm for embedded vision workloads. The project implements and benchmarks
distributed versions of core algorithms, including the 2D Fast Fourier Transform (FFT) for frequencydomain filtering and edge detection, against a baseline GPU implementation. The work demonstrates
the practical application of Distributed Memory MIMD architectures, emphasizing resilience,
elasticity, and the management of communication overhead in a resource-constrained edge environment.
Contents
1. Motivation: Scale-Up vs. Scale-Out 2
2. Theoretical Framework & Algorithmic Foundation 2
2.1 Parallel Random Access Machine (PRAM) Model Analysis . . . . . . . . . . . . . . . . . . . . 2
2.2 Communication Cost and Asynchrony . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
2.3 Scalability and Amdahl’s Law . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.4 Distributed 2D Fast Fourier Transform (FFT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
Proposed Distributed 2D FFT Algorithm (Row-Column Decomposition) . . . . . . . . . . . . 3
Frequency-Domain Edge Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
Inverse 2D FFT (IFFT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
3. System Architecture 4
3.1 Physical Hardware Configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
Configuration 1: Network Mesh / Wireless (Primary Target) . . . . . . . . . . . . . . . . . . 4
Configuration 2: Standard Ethernet (Wired Reference) . . . . . . . . . . . . . . . . . . . . . . 5
Configuration 3: High-Density “Bramble” (Advanced) . . . . . . . . . . . . . . . . . . . . . . 5
3.2 Simulation and Development Environment (Milestone 0) . . . . . . . . . . . . . . . . . . . . . . 5
4. Data Strategy 5
4.1 Dataset Complexity Visualized . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
5. Implementation Roadmap 6
6. Evaluation & Success Metrics 6
7. Conclusion 7
8. References 7
1
1. Motivation: Scale-Up vs. Scale-Out
The exponential growth of visual data from edge devices (surveillance, IoT, autonomous systems) has
outpaced the capabilities of single, centralized processors. While Moore’s Law provided decades of
performance gains, fundamental thermal and power density limits are now being faced. Parallel computing
is no longer an option but a necessity. This project empirically evaluates two dominant parallelization
strategies for embedded image processing:
• Scale-Up (Vertical Scaling): This approach enhances a single node’s computational capacity,
typically via specialized hardware like a Graphics Processing Unit (GPU). GPUs excel at data-parallel
tasks due to their massively parallel architecture (thousands of cores) and high memory bandwidth.
• Scale-Out (Horizontal Scaling): This approach aggregates multiple, often simpler and lower-power,
computing nodes into a cluster. It trades raw single-node performance for systemic advantages like
fault tolerance and elastic scalability.
Core Hypothesis: For a class of image processing algorithms amenable to decomposition, a carefully
engineered cluster of low-power ARM devices (Raspberry Pi 4) can achieve competitive throughput to
a mid-range GPU, while providing superior resilience (tolerating node failure) and elasticity (allowing
dynamic addition/removal of nodes). The primary challenge is to minimize communication overhead, which
is the principal adversary of distributed performance.
2. Theoretical Framework & Algorithmic Foundation
The system design is grounded in formal models from parallel and distributed computing. This section
expands the mathematical and algorithmic underpinnings.
2.1 Parallel Random Access Machine (PRAM) Model Analysis
The PRAM model abstracts away communication costs, allowing reasoning about the inherent parallelism
of algorithms. An image is modeled as a 2D matrix 𝐼 of size 𝑀 × 𝑁 pixels. For a pixel-wise operation (e.g.,
brightness adjustment, grayscale conversion), each of the 𝑃 processors can independently process 𝑀𝑁
𝑃
pixels.
• Theoretical Speedup: For an embarrassingly parallel operation of complexity 𝑂(𝑀𝑁), the ideal
speedup on 𝑃 processors is 𝑆𝑖𝑑𝑒𝑎𝑙(𝑃) = 𝑃.
• CREW Model Application: The distributed FFT and filter application uses a Concurrent Read,
Exclusive Write (CREW) PRAM strategy during the computation phase. All workers can read
the distributed image data concurrently, but writes to output buffers are coordinated to prevent race
conditions, often managed via master aggregation or careful buffer assignment.
2.2 Communication Cost and Asynchrony
In a physical distributed system, the interconnection network is a finite resource. The time to transfer a
message of size 𝐿 bytes between two nodes can be modeled as:
𝑇𝑐𝑜𝑚𝑚 = 𝛼 +
𝐿
𝛽
where 𝛼 is the latency (setup time per message) and 𝛽 is the bandwidth (bytes per second). For small
messages, latency dominates; for large image data, bandwidth is critical.
• Head-of-Line Blocking: In a synchronous system, a large data transfer (e.g., an image slice) can
block subsequent critical control messages (e.g., a “stop” signal), leading to deadlock or severe delays.
• Proposed Mitigation: Non-blocking point-to-point communication is implemented using
MPI_Isend and MPI_Irecv. This allows the computation thread to proceed while data is in transit,
effectively hiding communication latency. Control signals are sent via dedicated, high-priority channels
or asynchronously checked tags within non-blocking receives.
2
2.3 Scalability and Amdahl’s Law
The overall speedup of a parallel program is limited by its sequential fraction. Amdahl’s Law provides a
strict upper bound:
𝑆(𝑃 ) = 1
(1 − 𝑓) + 𝑓
𝑃
where 𝑓 is the parallelizable fraction of the program, and 𝑃 is the number of processors.
For the image processing pipeline:
1. Parallel Fraction (𝑓): The core pixel/row/column operations (FFT, filtering) are highly
parallelizable. The parallelizable fraction is estimated at 𝑓 ≈ 0.95 for the computational kernel.
2. Sequential Fraction (1 − 𝑓): This includes I/O (loading the image on the master), initial data
distribution (MPI_Scatter), final result collection (MPI_Gather), and any necessary serial operations
like matrix transposition on the master.
3. Theoretical Limit: Even with infinite workers (𝑃 → ∞), speedup is capped at 𝑆𝑚𝑎𝑥 =
1
1−𝑓 = 20. For
𝑃 = 6, the theoretical maximum is 𝑆(6) ≈ 5.1. The target of >4.5× speedup is thus ambitious,
requiring the sequential overhead to be minimized to less than 5%.
2.4 Distributed 2D Fast Fourier Transform (FFT)
The FFT is the cornerstone of frequency-domain image processing. The 2D DFT of an 𝑀 ×𝑁 image 𝑥[𝑚, 𝑛]
is defined as:
𝑋𝑘,𝑙 =
𝑀−1
∑
𝑚=0
𝑁−1
∑
𝑛=0
𝑥𝑚,𝑛 ⋅ 𝑒−𝑗2𝜋( 𝑘𝑚
𝑀 +
𝑙𝑛
𝑁 )
2D Separability: The equation can be decomposed, revealing that the 2D FFT is equivalent to applying
a 1D FFT to all rows, followed by a 1D FFT to all columns (or vice-versa):
𝑋𝑘,𝑙 =
𝑀−1
∑
𝑚=0 [
𝑁−1
∑
𝑛=0
𝑥𝑚,𝑛 ⋅ 𝑒−𝑗2𝜋 𝑙𝑛
𝑁 ] ⋅ 𝑒−𝑗2𝜋 𝑘𝑚
𝑀
Let 𝐺𝑚,𝑙 = FFT𝑁
1
(𝑥𝑚,∶), then 𝑋𝑘,𝑙 = FFT𝑀
1
(𝐺∶,𝑙).
Complexity Reduction: The direct 2D DFT requires 𝑂(𝑀2𝑁2
) operations. The separable 2D FFT
reduces this to 𝑂(𝑀𝑁 log2
(𝑀𝑁)). For 𝑀 = 𝑁 = 1024, this is a reduction from ∼ 1.1 × 1012 to ∼ 2.1 × 107
operations—a ~52,000× theoretical speedup before parallelism.
Proposed Distributed 2D FFT Algorithm (Row-Column Decomposition)
1. Data Distribution & Row-FFT (Parallel Phase):
• The master node partitions the 𝑀 ×𝑁 image into 𝑃 contiguous row blocks, each of size ≈
𝑀
𝑃 ×𝑁.
• Using MPI_Scatterv (to handle non-divisible 𝑀), the master sends one row block to each of the
𝑃 worker nodes.
• Each worker computes a 1D FFT on every row of its local block. Complexity per worker:
𝑂(𝑀
𝑃 𝑁 log2 𝑁).
2. Global Matrix Transposition (Communication Phase):
• The results of the row-FFTs are distributed by rows. The column-FFT requires data grouped by
columns.
• The master gathers all row-FFT blocks via MPI_Gatherv.
• The master performs an in-memory matrix transposition, converting the 𝑀 ×𝑁 matrix from
row-major to column-major ordering. Complexity: 𝑂(𝑀𝑁) memory operations.
3. Column-FFT & Finalization (Parallel Phase):
• The master redistributes the transposed matrix (now effectively column blocks) using another
MPI_Scatterv.
3
• Each worker computes a 1D FFT on every “row” of its new block (which corresponds to columns
of the original image).
• The master gathers the final results, forming the complete 2D FFT 𝑋𝑘,𝑙.
Communication Analysis: This algorithm requires two all-to-one (gather) and one one-to-all (scatter)
operations of size 𝑂(𝑀𝑁/𝑃 ) per node, plus the transpose on the master. The total communication volume
is 𝑂(𝑀𝑁).
Frequency-Domain Edge Detection
After obtaining the 2D FFT 𝑋𝑘,𝑙, a filter is applied to attenuate low frequencies (smooth regions) and
amplify high frequencies (edges). A Gaussian High-Pass Filter (GHPF) is used for its smooth transition,
reducing ringing artifacts:
𝐻(𝑘, 𝑙) = 1 − 𝑒−
𝐷2(𝑘,𝑙)
2𝐷2
0
where:
• 𝐷(𝑘, 𝑙) = √(𝑘 − 𝑀/2)2 + (𝑙 − 𝑁/2)2 is the distance from the center (DC component) in the frequency
domain, assuming the FFT output has been shifted.
• 𝐷0
is the cutoff frequency, controlling the severity of the filter. A smaller 𝐷0
removes more lowfrequency information, leaving only the sharpest edges.
The filtered frequency domain is: 𝑋filtered[𝑘, 𝑙] = 𝑋𝑘,𝑙 ⋅ 𝐻(𝑘, 𝑙).
Inverse 2D FFT (IFFT)
To return to the spatial domain and obtain the edge-enhanced image, the inverse 2D FFT is applied. The
2D IDFT is defined as:
𝑥𝑚,𝑛 =
1
𝑀𝑁
𝑀−1
∑
𝑘=0
𝑁−1
∑
𝑙=0
𝑋𝑘,𝑙 ⋅ 𝑒𝑗2𝜋( 𝑘𝑚
𝑀 +
𝑙𝑛
𝑁 )
The same separable property holds. The distributed algorithm mirrors the forward FFT process:
1. Columns of 𝑋filtered are scattered for 1D IFFT.
2. Results are gathered, transposed, and scattered as rows for the second 1D IFFT.
3. Results are gathered and the 1
𝑀𝑁 scaling factor is applied.
The final output 𝑥edge[𝑚, 𝑛] highlights the edges present in the original image.
3. System Architecture
3.1 Physical Hardware Configurations
Based on industry standards and official Raspberry Pi cluster guidelines, three potential hardware
architectures are proposed:
Configuration 1: Network Mesh / Wireless (Primary Target)
A decentralized approach emphasizing mobility and zero-infrastructure cost.
• Nodes: 6× Raspberry Pi 4
• Network: WiFi (802.11ac) Mesh
• Power: Individual USB-C supplies or portable power banks
• Advantages: Zero network cabling, highly portable
• Research Value: Excellent testbed for fault tolerance algorithms due to natural latency and packet
loss
4
Configuration 2: Standard Ethernet (Wired Reference)
The baseline robust cluster setup for high-throughput benchmarking.
• Nodes: 6× Raspberry Pi 4 (1 Master, 5 Workers)
• Network: 8-Port Gigabit Switch
• Storage: Individual 32GB MicroSD Cards
• Advantages: High reliability, lower latency, easier debugging
Configuration 3: High-Density “Bramble” (Advanced)
A professional, data-center style deployment.
• Nodes: 6× Raspberry Pi 4 with PoE+ HATs
• Network/Power: Single PoE+ Switch
• Storage: Diskless (PXE Netboot from Master’s SSD)
• Advantages: Minimal cabling, centralized management
3.2 Simulation and Development Environment (Milestone 0)
Prior to physical assembly, a virtualized cluster is created to develop and validate the software stack.
• Containerization: Each node is represented by a Docker container using the python:3.9-slim base
image, tailored to mimic Raspberry Pi OS Lite.
• Architecture Emulation: While development laptops are x86_64, builds and tests use the
linux/arm64 platform flag (--platform linux/arm64) to ensure binary compatibility.
• Resource Constraints: Containers are limited to 1 CPU core and 2GB of RAM to accurately
reflect the resource constraints of a Raspberry Pi 4, forcing efficient algorithm design.
4. Data Strategy
Phase Dataset Characteristics Objective
Sim/Debug CIFAR-10 32×32 px images Functional Testing: Verification of
network protocols and MPI message
passing
Benchmark Tiny ImageNet 64×64 px Stress Testing: Saturation of
interconnect and testing of load balancing
algorithms
Final Demo COCO (Val) 4K / HD images Real-World Demo: Complex object
detection (e.g., traffic monitoring)
4.1 Dataset Complexity Visualized
5
CIFAR-10 (32×32) Tiny ImageNet (64×64) COCO (HD / Real World)
Low Bandwidth / Debug High Traffic / Stress Compute Intensive / Demo
5. Implementation Roadmap
A phased approach ensures incremental progress and manageable complexity.
1. Milestone 0: Virtual Cluster & Toolchain
• A Docker-based simulation environment is established.
• The cross-compilation/emulation toolchain for ARM is set up.
• A basic “Hello World” MPI program (mpi4py) is implemented to verify communication.
2. Milestone 1: Foundations of Concurrency
• On a single Pi (multi-core): A multi-threaded image filter (e.g., Sobel edge detector) is
implemented using Python’s threading with locks or multiprocessing pools.
• Speedup on 4 cores vs. single core is benchmarked, observing the effects of Python’s GIL.
3. Milestone 2: Theory & Shared Memory Algorithms
• Formal PRAM analysis of chosen algorithms is conducted.
• Inter-process communication (IPC) is implemented using shared memory (/dev/shm) for
multi-process programs on a single node, comparing performance to threading.
4. Milestone 3: Physical Build & Distributed MPI
• The chosen physical hardware configuration is assembled.
• The cluster is configured: SSH key-based authentication, hostfile setup, network troubleshooting.
• A distributed image processing task is implemented using MPI’s collective operations (Scatter,
Gather), e.g., distributed histogram equalization.
5. Milestone 4: Optimization & Resilience
• Overlap Communication/Computation: Code is refactored to use MPI_Isend/MPI_Irecv.
• The Bully Algorithm for leader election is implemented to provide master failover.
• Benchmarking: Comprehensive performance profiling is conducted across all hardware
configurations.
6. Milestone 5: Integration & Final Demo
• The distributed 2D FFT pipeline is integrated into a cohesive application.
• Final demonstrations are prepared: live processing of COCO images, simulated node failure and
recovery, performance comparison (Cluster vs. single Pi vs. baseline GPU).
• The final report and analysis of results against theoretical predictions is completed.
6. Evaluation & Success Metrics
Quantifiable metrics are essential for validating the design and implementation.
6
Metric Target Description
Speedup (𝑆𝑝
) > 4.5× Measured on 6 nodes vs. single node
Parallel Efficiency (𝐸𝑝
) > 85% Indicates low communication overhead
Throughput > 40 FPS Real-time object detection rate on full cluster
Failover Latency < 3 sec Time to elect new leader and resume processing
7. Conclusion
This proposal outlines a comprehensive project to build and analyze an Intelligent Edge Distributed Image
Processing System. By grounding the work in parallel computing theory (PRAM, Amdahl’s Law) and
tackling a non-trivial algorithmic problem (Distributed 2D FFT), the project moves beyond simple task
farming. The investigation of multiple hardware topologies yields insights into the practical trade-offs
between performance, robustness, and deployability in edge computing scenarios. The final system serves
as a tangible demonstration of scale-out principles and a flexible testbed for future distributed algorithms
research.
8. References
• Raspberry Pi Foundation. (n.d.). Raspberry Pi documentation. Retrieved from https://www.raspberr
ypi.com/documentation/
• Raspberry Pi Foundation. (n.d.). Build a Raspberry Pi cluster. Retrieved from https://www.raspberr
ypi.com/tutorials/cluster-raspberry-pi-tutorial/
• CSE-467 Parallel and Distributed Computing. (Modules 1–6). Course materials.
Note: This project proposal is a living document. Milestones, technical specifications, and target
metrics may be refined based on empirical findings and implementation challenges encountered
during the project lifecycle.
