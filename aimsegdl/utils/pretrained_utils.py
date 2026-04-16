import importlib.resources


def list_pretrained_weights():
    pkg = importlib.resources.files("aimsegdl.weights")
    return [p.name[:-4] for p in pkg.iterdir() if p.name.endswith(".pth")]
