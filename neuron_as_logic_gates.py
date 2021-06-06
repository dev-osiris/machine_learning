import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def plot_sigmoid():
    vals = np.linspace(-10, 10, num=100, dtype=np.float32)
    activation = sigmoid(vals)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(vals, activation)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.yticks()
    plt.ylim([-0.5, 1.5])
    plt.show()

def logic_gate(w1, w2, b):
    return lambda x1, x2: sigmoid(w1 * x1 + w2 * x2 + b)

def test(gate):
    for a, b in (0, 0), (0, 1), (1, 0), (1, 1):
        print(f"{a} {b} | {int(np.round(gate(a, b)))}")

or_gate = logic_gate(20, 20, -10)
and_gate = logic_gate(11, 10, -20)
nand_gate = logic_gate(-10, -10, 20)
#for AND gate w1 = 11, w2 = 10, b = -20
#for NOR gate w1 = -20, w2 = -20, b = 10
#for NAND gate w1 = -10, w2 = -10, b = 20

def xor_gate(a, b):
    c = or_gate(a, b)
    d = nand_gate(a, b)
    return and_gate(c, d)

test(xor_gate)