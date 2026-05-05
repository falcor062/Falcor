import os
import os.path as osp
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image


import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



import os
import numpy as np
import pandas as pd

root = r"D:\object-classification\datasets\nuswide"

# ---- load concepts ----
concepts_path = os.path.join(root, "ConceptsList", "Concepts81.txt")
with open(concepts_path) as f:
    concepts = [c.strip() for c in f.readlines()]

# ---- load image lists ----
def load_list(path):
    with open(path) as f:
        return [l.strip() for l in f.readlines()]

train_imgs = load_list(os.path.join(root, "ImageList", "TrainImagelist.txt"))
test_imgs  = load_list(os.path.join(root, "ImageList", "TestImagelist.txt"))

all_imgs = train_imgs + test_imgs
split_map = {img: "train" for img in train_imgs}
split_map.update({img: "val" for img in test_imgs})

# ---- load labels ----
labels = []
for concept in concepts:
    path_train = os.path.join(root, "Groundtruth", "TrainTestLabels", f"Labels_{concept}_Train.txt")
    path_test  = os.path.join(root, "Groundtruth", "TrainTestLabels", f"Labels_{concept}_Test.txt")

    with open(path_train) as f:
        train_vals = [int(x.strip()) for x in f.readlines()]
    with open(path_test) as f:
        test_vals = [int(x.strip()) for x in f.readlines()]

    labels.append(train_vals + test_vals)

labels = np.array(labels).T  # [N, C]

# ---- build CSV ----
rows = []
for i, img in enumerate(all_imgs):
    active = [concepts[j] for j in range(len(concepts)) if labels[i, j] == 1]
    rows.append({
        "filename": img,
        "label": str(active),
        "split": split_map[img]
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(root, "nus_wid_data.csv"), index=False)

print("Saved nus_wid_data.csv")
class NUSWide(Dataset):
    def __init__(self, root, image_root=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.image_root = image_root if image_root is not None else osp.join(root, "images")
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.image_list, self.labels = self._load_split()

        assert len(self.image_list) == len(self.labels), (
            f"Image count {len(self.image_list)} != label count {len(self.labels)}"
        )

        self.num_classes = self.labels.shape[1]
        self.itemlist = self._build_items()


    def _load_split(self):
        if self.train:
            img_list_path = osp.join(self.root, "database_img.txt")
            labels_path = osp.join(self.root, "database_label_onehot.txt")
        else:
            img_list_path = osp.join(self.root, "test_img.txt")
            labels_path = osp.join(self.root, "test_label_onehot.txt")

        if not osp.exists(img_list_path):
            raise FileNotFoundError(f"Missing image list file: {img_list_path}")
        if not osp.exists(labels_path):
            raise FileNotFoundError(f"Missing label file: {labels_path}")

        with open(img_list_path, "r", encoding="utf-8", errors="ignore") as f:
            image_list = [line.strip() for line in f if line.strip()]

        labels = np.loadtxt(labels_path, dtype=np.float32)

        if labels.ndim == 1:
            labels = labels[:, None]

        return image_list, labels

    def _build_items(self):
        items = []
        missing = 0
        shown_missing = 0

        for rel_img_path, label_vec in zip(self.image_list, self.labels):
            rel_img_path = rel_img_path.strip().replace("/", os.sep).replace("\\", os.sep)

            # Remove duplicated leading folder names if needed
            if rel_img_path.startswith("images" + os.sep):
                rel_img_path = rel_img_path[len("images" + os.sep):]

            img_path = osp.normpath(osp.join(self.image_root, rel_img_path))

            if not osp.exists(img_path):
                missing += 1
                if shown_missing < 5:
                    print(f"[NUSWide] Missing example: {img_path}")
                    shown_missing += 1
                continue

            items.append((img_path, label_vec))

        if not items:
            raise RuntimeError(
                f"No valid images found under {self.image_root}. "
                "Check image_root and the relative paths in the image list file."
            )

        if missing > 0:
            print(f"[NUSWide] Skipped {missing} missing images.")

        return items

    def __len__(self):
        return len(self.itemlist)

    def __getitem__(self, index):
        imgpath, labels = self.itemlist[index]

        img = Image.open(imgpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels

# class NUSWide(Dataset):
#     def __init__(self, root, train=True, transform=None, target_transform=None ) -> None:
#         self.img_dir = root
#         self.csv_path = os.path.join(root, "nus_wid_data.csv")
#
#         if train:
#             self.split = "train"
#         else:
#             self.split = "val"
#
#         self.transform = transform
#         self.target_transform = target_transform
#
#         self.itemlist, self.num_classes = self.preprocess()
#
#     def preprocess(self):
#         # read csv file
#         df = pd.read_csv(self.csv_path)
#         labels_col = df['label']
#         labels_list_all = []
#         for item in labels_col:
#             i_labellist = str_to_list(item)
#             labels_list_all.extend(i_labellist)
#         labels_list_all = sorted(list(set(labels_list_all)))
#         labels_map = {labelname:idx for idx, labelname in enumerate(labels_list_all)}
#         length = len(labels_list_all)
#
#         # generate itemlist
#         res = []
#         for index, row in df.iterrows():
#             split_name = row[2]
#             if split_name != self.split and self.split != 'all':
#                 continue
#             filename = row[0]
#             imgpath = osp.join(self.img_dir, filename)
#             label = [labels_map[i] for i in str_to_list(row[1])]
#             label_np = np.zeros(length, dtype='float32')
#             for idd in label:
#                 label_np[idd] = 1.0
#             res.append((imgpath, label_np))
#
#         return res, length
#
#     def __len__(self) -> int:
#         return len(self.itemlist)
#
#     def __getitem__(self, index: int):
#         imgpath, labels = self.itemlist[index]
#
#         img = Image.open(imgpath).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             labels = self.target_transform(labels)
#         return img, labels


def str_to_list(text):
    """
    input: "['clouds', 'sky']" (str)
    output: ['clouds', 'sky'] (list)
    """
    # res = []
    res = [i.strip('[]\'\"\n ') for i in text.split(',')]
    return res
