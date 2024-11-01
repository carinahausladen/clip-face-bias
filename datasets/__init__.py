from datasets.fairface import FairfaceDataset
from datasets.causalface import CausalFaceDataset
from datasets.utk_face import UTKFaceDataset


def get_dataset(name: str, **kwargs):

    if name.endswith("_rm_bg"):
        name = name.replace("_rm_bg", "")
        kwargs["remove_background"] = True

    if name == "fairface":
        return FairfaceDataset(**kwargs)
    elif name == "causalface" or name == "causalface_age":
        return CausalFaceDataset(subset="age", **kwargs)
    elif name == "causalface_lighting":
        return CausalFaceDataset(subset="lighting", **kwargs)
    elif name == "causalface_pose":
        return CausalFaceDataset(subset="pose", **kwargs)
    elif name == "causalface_smiling":
        return CausalFaceDataset(subset="smiling", **kwargs)
    elif name == "causalface_brightness":
        return CausalFaceDataset(subset="brightness", **kwargs)
    elif name == "utkface":
        return UTKFaceDataset()
    else:
        raise NotImplementedError(f"Unknown dataset {name}")