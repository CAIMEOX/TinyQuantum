from functools import reduce
from scipy.linalg import kron
import numpy as np

spin_up = np.array([[1,0]]).T
spin_down = np.array([[0,1]]).T

bit = [spin_up, spin_down]

# string to basis
to_basis = lambda string: np.matrix(reduce(lambda x, y: kron(bit[int(y)], x), string[::-1], np.array([[1]])))

# hilbert space
def hilbert_space(bits=2):
    for i in range(1 << bits):
        binary = bin(i)[2:] # remove 0b
        zeros = bits - len(binary)
        yield '0' * zeros + binary

# Construct superposition of wave function.
wave_fn = lambda coef, seqs: np.matrix(sum(map(lambda a, seq: a * to_basis(seq), coef, seqs)))
# Project function onto a direction.
project = lambda wave_func, direction: wave_func.H * direction
# Validate amplitudes
validate = lambda amplitudes: np.equal(np.linalg.norm(amplitudes),1)
# decompose a wave function into amplitudes and directions
def decompose(wave_func):
    bits = int(np.log2(len(wave_func)))
    amplitudes = []
    direct_str = []
    for seq in hilbert_space(bits):
        direct = to_basis(seq)
        amp = project(wave_func, direct).A1[0]
        if np.linalg.norm(amp) != 0:
            amplitudes.append(amp)
            direct_str.append(seq)
    return amplitudes, direct_str

# Format wave function as a string | LaTeX
format_wave_fn = lambda wf: " + ".join(f"{c} | {s} >" for c, s in zip(*decompose(wf)))

# Quantum gates
# Pauli matrix
I = np.matrix([[1,0],[0,1]])
X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])
Z = np.matrix([[1,0],[0,-1]])
# Hadamard gate
H = np.matrix([[1,1],[1,-1]])/np.sqrt(2)
# Multi-qubit gates
CNOT = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
SWAP = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
print(validate([1,1]/np.sqrt(2)))