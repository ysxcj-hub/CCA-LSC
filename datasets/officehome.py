import os
from typing import Optional, Tuple, Any

from tllib.vision.datasets.imagelist import ImageList
from tllib.vision.datasets._util import download as download_data, check_exits


class OfficeHome(ImageList):
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    ]
    # image_list = {
    #     "Ar": "image_list/Art.txt",
    #     "Cl": "image_list/Clipart.txt",
    #     "Pr": "image_list/Product.txt",
    #     "Rw": "image_list/Real_World.txt",
    # }

    image_list = {
        # "Ar": "image_list/txt/Art.txt",
        "Cl": "Clipart",
        "Pr": "Product",
        "Rw": "Real_World",
    }
    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

    def __init__(self, root: str, task: str,split: Optional[str]='RS',mask:Optional[bool]=False, download: Optional[bool] = False, **kwargs):
        # assert task in self.image_list
        # data_list_file = os.path.join(root, self.image_list[task])
        assert task in self.image_list
        assert split in ['RS', 'UT']
        if not mask:
            data_list_file = os.path.join(root,"txt", "{}_{}.txt".format(self.image_list[task], split))
        else:
            data_list_file = os.path.join(root, "txt", "{}_{}_mask.txt".format(self.image_list[task], split))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(OfficeHome, self).__init__(root, OfficeHome.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

    # def __getitem__(self, index: int) -> Tuple[Any, int]:
    #     """
    #     Args:
    #         index (int): Index
    #
    #     return (tuple): (image, target) where target is index of the target class.
    #     """
    #     path, target = self.samples[index]
    #     img = self.loader(path).convert(self.mode)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None and target is not None:
    #         target = self.target_transform(target)
    #     return img, target

    @classmethod
    def get_classes(self):
        return OfficeHome.CLASSES


