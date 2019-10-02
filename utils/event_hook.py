def hook_bert_layer_output(bert_model, layer_idx, hook):
    bert_model.encoder.layer[layer_idx].attention.output.register_forward_hook(hook)


def hook_bert_layer_attn(bert_model, layer_idx, hook):
    bert_model.encoder.layer[layer_idx].attention.self.register_forward_hook(hook)


class BundleAveragingHook(object):
    def __init__(self):
        self.bundle = None
        self.data = []

    def __call__(self, module, tensor_in, tensor_out):
        if not self.bundle:
            return
        out = tensor_out[0] if isinstance(tensor_out, tuple) else tensor_out
        out = self.bundle.mean(out.detach()).cpu()
        self.data.append(out)