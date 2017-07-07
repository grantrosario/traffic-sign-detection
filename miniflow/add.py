from miniflow import *

# Define two 'input' nodes.
x, y, z = Input(), Input(), Input()

# Define an 'Add' node, the two above 'Input' nodes being the input.
f = Add(x, y, z)

# The value of 'x' and 'y' will be set to 10 and 20 respectively
feed_dict = {x: 10, y: 20, z: 15}

# Sort the nodes with topological sort
sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
