>....                                                                                                                                                             
length_in = None
for inp in base_inputs:
    if getattr(inp, 'name', '').split(':')[0] == 'length_input':
        length_in = inp
        break
if length_in is not None:
    lengths_flat = keras.layers.Lambda(lambda x: K.squeeze(K.cast(x, 'int32'), axis=-1), name='lengths_flat')(length_in)
else:
    lengths_flat = keras.layers.Lambda(lambda z: K.full((K.shape(z)[0],), K.shape(z)[1], dtype='int32'), name='lengths_flat')(feats)

def _mk_mask(args):
    z, ln = args
    T = K.shape(z)[1]
    t_idx = K.arange(T)[None, :]
    ln_e = K.expand_dims(K.cast(ln, 'int32'), -1)
    return K.less(t_idx, ln_e)
mask_bt = keras.layers.Lambda(_mk_mask, name='token_mask')([feats, lengths_flat])

decoded, potentials, lengths_crf, trans = wrap.crf(feats, mask=mask_bt)
loss_vec = CRFNLLHead(name='crf_log_likelihood_output')([potentials, labels_in, lengths_flat, trans])
decoded_named = keras.layers.Lambda(lambda z: K.stop_gradient(z), name='decoded_output')(decoded)

outs = [decoded_named, loss_vec]
all_inputs = [tokens_in] + base_inputs + [labels_in]

inputs_set = set(tf.nest.flatten(all_inputs))
outputs_list = tf.nest.flatten(outs)
reachable_inputs = set()
visited = set()
q = deque(outputs_list)

while q:
    t = q.popleft()
    if id(t) in visited:
        continue
    visited.add(id(t))
    hist = getattr(t, 'keras_history', None) or getattr(t, '_keras_history', None)
    if not hist:
        continue
    layer = getattr(hist, 'operation', None) or getattr(hist, 'layer', None)
    if layer is None:
        continue
    inps = tf.nest.flatten(layer.input)
    for inp in inps:
        if inp in inputs_set:
            reachable_inputs.add(inp)
        q.append(inp)

missing = inputs_set - reachable_inputs
name = lambda x: getattr(x, 'name', '').split(':')[0]
print('All inputs:', [name(x) for x in inputs_set])
print('Reachable inputs:', [name(x) for x in reachable_inputs])
print('UNCONNECTED inputs:', [name(x) for x in missing])
PY
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1757297580.890502 4105017 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1757297580.893116 4105017 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
/home/m_thing/.local/share/mamba/envs/kaggle/lib/python3.11/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
BidLSTM_CRF
All inputs: ['char_input', 'tokens', 'labels', 'length_input', 'word_input']
Reachable inputs: ['word_input', 'length_input', 'labels', 'char_input']
UNCONNECTED inputs: ['tokens']