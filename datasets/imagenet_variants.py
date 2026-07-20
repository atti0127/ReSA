import os

from .imagenet import imagenet_classes
from .utils import Datum, DatasetBase, listdir_nohidden


TO_BE_IGNORED = {"README.txt"}
template = ["a photo of a {}."]


def read_classnames(text_file):
    """Read ImageNet-style ``classnames.txt`` files.

    The expected format is ``<folder_or_wnid> <class name>`` per line.  If the
    file is absent, fall back to numeric ImageNet labels so ImageNetV2 still
    works in datasets that only store folders ``0`` ... ``999``.
    """

    if not os.path.exists(text_file):
        return {str(label): cname for label, cname in enumerate(imagenet_classes)}

    classnames = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames


class _ImageNetVariant(DatasetBase):
    dataset_dir = ""
    image_subdir = ""

    def __init__(self, root, num_shots=None):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.image_subdir)
        self.template = template

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)
        test = self.read_data(classnames)

        # These datasets are evaluation-only in the cross-dataset/DG protocol.
        # Reusing test as train_x lets DatasetBase infer class names without
        # forcing a non-existent training split.
        super().__init__(train_x=test, val=test, test=test)


class ImageNetV2(_ImageNetVariant):
    """ImageNetV2 matched-frequency validation set."""

    dataset_dir = "imagenetv2"
    image_subdir = "imagenetv2-matched-frequency-format-val"

    def read_data(self, classnames):
        folders = [str(label) for label in range(1000)]
        items = []

        for label, folder in enumerate(folders):
            class_dir = os.path.join(self.image_dir, folder)
            imnames = listdir_nohidden(class_dir, sort=True)
            classname = classnames.get(folder, imagenet_classes[label])
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                items.append(Datum(
                    impath=impath,
                    label=label,
                    classname=classname,
                ))

        return items


class _FolderSubsetImageNetVariant(_ImageNetVariant):
    """ImageNet variants whose folders are the evaluated class subset."""

    def read_data(self, classnames):
        folders = listdir_nohidden(self.image_dir, sort=True)
        folders = [folder for folder in folders if folder not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            class_dir = os.path.join(self.image_dir, folder)
            imnames = listdir_nohidden(class_dir, sort=True)
            classname = classnames.get(folder, folder.replace("_", " "))
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                items.append(Datum(
                    impath=impath,
                    label=label,
                    classname=classname,
                ))

        return items


class ImageNetA(_FolderSubsetImageNetVariant):
    """ImageNet-A(dversarial), evaluation-only."""

    dataset_dir = "imagenet-adversarial"
    image_subdir = "imagenet-a"


class ImageNetR(_FolderSubsetImageNetVariant):
    """ImageNet-R(endition), evaluation-only."""

    dataset_dir = "imagenet-rendition"
    image_subdir = "imagenet-r"


class ImageNetSketch(_FolderSubsetImageNetVariant):
    """ImageNet-Sketch, evaluation-only."""

    dataset_dir = "imagenet-sketch"
    image_subdir = "images"
