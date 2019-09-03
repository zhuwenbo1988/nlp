import tensorflow as tf
import numpy as np

vocab = [line.split()[0] for line in open('qq_data/char_based_vocab', 'r').read().splitlines()]
token2idx = {token: idx for idx, token in enumerate(vocab)}

def gen_model_input(s):
    tokens = s.split()
    x = [token2idx.get(t, token2idx["<unk>"]) for t in tokens]
    x_seqlen = len(x)
    return (x, x_seqlen, s)

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    # 需要导出
    output_graph_path = 'tsf.pb'

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        input_ids = sess.graph.get_tensor_by_name("input_ids:0")
        input_len = sess.graph.get_tensor_by_name("input_len:0")

        output = sess.graph.get_tensor_by_name("encoder/num_blocks_0/positionwise_feedforward/ln/add_1:0")
        avg_output = sess.graph.get_tensor_by_name("avg_vector:0")
 
        print(input_ids)
        print(input_len)
        print(output)
        print(avg_output)

        all_v = []
        path = 'sample_infer_data/dazhong.char'
        for line in open(path):
            s = line.strip()
            
            x, x_seqlen, s = gen_model_input(s)

            feed_dict = {input_ids:np.array([x]), input_len:np.array([x_seqlen])}

            out = sess.run(avg_output, feed_dict=feed_dict)
            v = out[0]
            all_v.append(v)
        np.save("tsf_vector.npy",all_v)        
