import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class PairedDataset(BaseDataset):
    """A dataset class for paired images stored in separate A/ and B/ subfolders.

    It expects the directory structure:
        /path/to/data/{phase}/A/  -- input domain images
        /path/to/data/{phase}/B/  -- target domain images

    Images are paired by sorted filename order. A and B can have different
    resolutions (e.g., 256x256 input and 1024x1024 target for super-resolution).
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B")
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert len(self.A_paths) == len(self.B_paths), \
            f"A ({len(self.A_paths)}) and B ({len(self.B_paths)}) must have the same number of images"
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) -- an image in the input domain
            B (tensor) -- its corresponding image in the target domain
            A_paths (str) -- image path for A
            B_paths (str) -- image path for B
        """
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = Image.open(A_path).convert("RGB")
        B = Image.open(B_path).convert("RGB")

        # Apply transforms independently (A and B may differ in resolution),
        # but share the same random flip for spatial consistency.
        A_params = get_params(self.opt, A.size)
        B_params = get_params(self.opt, B.size)
        B_params["flip"] = A_params["flip"]  # sync flip

        A_transform = get_transform(self.opt, A_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, B_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
