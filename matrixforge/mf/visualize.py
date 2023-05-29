import matplotlib.pyplot as plt

def model_visualize(layer_sizes,neuron_size='',color=''):
        num_layers = len(layer_sizes)
        layer_scale = len(layer_sizes)/10
        layer_offset = len(layer_sizes)/2
        if num_layers == 1:
          raise ValueError("Number of layer must contain two")
        layer_positions = []
        for l in range(num_layers):
           num_nodes = layer_sizes[l]
           layer_y = linspace(0, 1, num_nodes + 2)[1:-1]
           layer_y *= layer_scale
           layer_y += 0.1
           layer_x = ones(num_nodes) * layer_offset
           layer_positions.append(column_stack((layer_x, layer_y)))
           layer_offset += 1
        connections = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        color_index = 0
        for l in range(num_layers - 1):
           positions1 = layer_positions[l]
           positions2 = layer_positions[l + 1]
           for i in range(positions1.shape[0]):
             for j in range(positions2.shape[0]):
                 connections.append(vstack((positions1[i], positions2[j])))
                 if color_index < len(colors):
                     color_index += 1
                 else:
                     color_index = 0
        fig, ax = plt.subplots()
        for pos in layer_positions:
         ax.scatter(pos[:, 0], pos[:, 1], color=color, s=neuron_size)
        for i in range(len(connections)):
         ax.plot(connections[i][:, 0], connections[i][:, 1], color=colors[i % len(colors)], linewidth=neuron_size/100)
         ax.axis('off')
        plt.show()