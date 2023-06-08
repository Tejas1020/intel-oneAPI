import tensorflow as tf
from tensorflow.python.framework import graph_util

#change the names of both files accordingly

# Load the trained TensorFlow model
loaded_model = tf.keras.models.load_model('/content/drive/MyDrive/Datasets/lyme_disease/Best_model.tf')

input_node_names = ['input_node']  # Replace with actual input node names
output_node_names = ['output_node']  # Replace with actual output node names

graph = tf.compat.v1.get_default_graph()

# Convert the graph to a GraphDef object
graph_def = graph.as_graph_def()

frozen_graph_def = graph_util.convert_variables_to_constants(
    sess=tf.compat.v1.Session(),
    input_graph_def=graph_def,
    output_node_names=output_node_names
)


output_dir = 'path_to_save_frozen_model'
output_file = 'frozen_model.pb'
output_path = output_dir + '/' + output_file

with tf.io.gfile.GFile(output_path, 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())

print('Frozen model saved to:', output_path)

output_graph_def_path = 'path_to_save_graph_def.pbtxt'
tf.train.write_graph(graph_def, '.', output_graph_def_path, as_text=True)
