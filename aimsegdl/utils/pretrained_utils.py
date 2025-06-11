import pkg_resources
import os

def list_pretrained_weights():
    return [os.path.splitext(f)[0] for f in pkg_resources.resource_listdir("aimsegdl.weights", "")
            if f.endswith(".pth")]
