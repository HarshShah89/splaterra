import torch, torch.nn.functional as F

class GateSWACollector:
    def __init__(self, model):
        self.model = model
        self.window_idx = 0
        self.records = []
        self.handles = []
        for i, proj in enumerate(model.ttt_gate_projs):
            self.handles.append(proj.register_forward_hook(self._make_hook("ttt", i)))
        for i, proj in enumerate(model.swa_gate_projs):
            self.handles.append(proj.register_forward_hook(self._make_hook("swa", i)))

    def _make_hook(self, kind, layer_idx):
        def hook(module, inp, out):
            gate = F.silu(out).abs().mean().item()
            self.records.append({"window": self.window_idx, "kind": kind,
                                  "layer": layer_idx, "gate_mean": gate})
        return hook

    def advance_window(self):
        self.window_idx += 1

    def remove(self):
        for h in self.handles: h.remove()