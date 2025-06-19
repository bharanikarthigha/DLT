# McCulloch-Pitts Neuron implementation

def mcp_neuron(inputs, weights, threshold):
    """
    inputs: list of binary inputs (0 or 1)
    weights: list of weights corresponding to inputs
    threshold: activation threshold
    returns: 0 or 1 (binary output)
    """
    total_input = sum(i * w for i, w in zip(inputs, weights))
    return 1 if total_input >= threshold else 0

# Logic functions using MCP neuron

def AND_gate(x1, x2):
    return mcp_neuron([x1, x2], weights=[1, 1], threshold=2)

def OR_gate(x1, x2):
    return mcp_neuron([x1, x2], weights=[1, 1], threshold=1)

def NOT_gate(x):
    # For NOT, we simulate a single input with negative weight
    return mcp_neuron([x], weights=[-1], threshold=0)

# Test the gates
print("AND Gate")
for a in [0, 1]:
    for b in [0, 1]:
        print(f"{a} AND {b} = {AND_gate(a, b)}")

print("\nOR Gate")
for a in [0, 1]:
    for b in [0, 1]:
        print(f"{a} OR {b} = {OR_gate(a, b)}")

print("\nNOT Gate")
for a in [0, 1]:
    print(f"NOT {a} = {NOT_gate(a)}")
