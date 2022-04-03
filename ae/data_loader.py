# Original code from https://github.com/araffin/robotics-rl-srl
# Authors: Antonin Raffin, René Traoré, Ashley Hill

import random
import time
from multiprocessing import Process, Queue

import cv2
import imgaug
import numpy as np
import torchvision.transforms.functional as vision_fn
from imgaug import augmenters as iaa
from imgaug.augmenters import Sometimes
from joblib import Parallel, delayed
from PIL import Image, ImageChops
from six.moves import queue

from ae.autoencoder import preprocess_image, preprocess_input


class CheckFliplrPostProcessor:
    def __init__(self):
        super().__init__()
        self.flipped = False

    def __call__(self, images, augmenter, parents):
        if "Fliplr" in augmenter.name:
            self.flipped = True
        return images


def get_image_augmenter() -> iaa.Sequential:
    """
    :return: Image Augmenter
    """
    return iaa.Sequential(
        [
            Sometimes(0.5, iaa.Fliplr(1)),
            # Add shadows (from https://github.com/OsamaMazhar/Random-Shadows-Highlights)
            Sometimes(0.3, RandomShadows(1.0)),
            # Sometimes(0.3, iaa.MultiplyBrightness((0.8, 1.2))),
            Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),
            Sometimes(0.5, iaa.MotionBlur(k=(3, 11), angle=(0, 360))),
            # Sometimes(0.5, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))),
            Sometimes(0.4, iaa.Add((-25, 25), per_channel=0.5)),
            # Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
            # Sometimes(0.2, iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.10), per_channel=0.5)),
            # 20% of the corresponding size of the height and width
            Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
            # Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 1.8), per_channel=0.5)),
            # Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True))
        ],
        random_order=True,
    )


# Adapted from https://github.com/OsamaMazhar/Random-Shadows-Highlights
class RandomShadows(iaa.meta.Augmenter):
    def __init__(
        self,
        p=0.5,
        high_ratio=(1, 2),
        low_ratio=(0.01, 0.5),
        left_low_ratio=(0.4, 0.6),
        left_high_ratio=(0, 0.2),
        right_low_ratio=(0.4, 0.6),
        right_high_ratio=(0, 0.2),
        seed=None,
        name=None,
    ):
        super().__init__(seed=seed, name=name)

        self.p = p
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.left_low_ratio = left_low_ratio
        self.left_high_ratio = left_high_ratio
        self.right_low_ratio = right_low_ratio
        self.right_high_ratio = right_high_ratio

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i in range(batch.nb_rows):
            if random.uniform(0, 1) < self.p:
                batch.images[i] = self.process(
                    batch.images[i],
                    self.high_ratio,
                    self.low_ratio,
                    self.left_low_ratio,
                    self.left_high_ratio,
                    self.right_low_ratio,
                    self.right_high_ratio,
                )
        return batch

    @staticmethod
    def process(
        img,
        high_ratio,
        low_ratio,
        left_low_ratio,
        left_high_ratio,
        right_low_ratio,
        right_high_ratio,
    ):

        img = Image.fromarray(img)
        w, h = img.size
        # h, w, c = img.shape
        high_bright_factor = random.uniform(high_ratio[0], high_ratio[1])
        low_bright_factor = random.uniform(low_ratio[0], low_ratio[1])

        left_low_factor = random.uniform(left_low_ratio[0] * h, left_low_ratio[1] * h)
        left_high_factor = random.uniform(left_high_ratio[0] * h, left_high_ratio[1] * h)
        right_low_factor = random.uniform(right_low_ratio[0] * h, right_low_ratio[1] * h)
        right_high_factor = random.uniform(right_high_ratio[0] * h, right_high_ratio[1] * h)

        tl = (0, left_high_factor)
        bl = (0, left_high_factor + left_low_factor)

        tr = (w, right_high_factor)
        br = (w, right_high_factor + right_low_factor)

        contour = np.array([tl, tr, br, bl], dtype=np.int32)

        mask = np.zeros([h, w, 3], np.uint8)
        cv2.fillPoly(mask, [contour], (255, 255, 255))
        inverted_mask = cv2.bitwise_not(mask)
        # we need to convert this cv2 masks to PIL images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # we skip the above convertion because our mask is just black and white
        mask_pil = Image.fromarray(mask)
        inverted_mask_pil = Image.fromarray(inverted_mask)

        low_brightness = vision_fn.adjust_brightness(img, low_bright_factor)
        low_brightness_masked = ImageChops.multiply(low_brightness, mask_pil)
        high_brightness = vision_fn.adjust_brightness(img, high_bright_factor)
        high_brightness_masked = ImageChops.multiply(high_brightness, inverted_mask_pil)

        return np.array(ImageChops.add(low_brightness_masked, high_brightness_masked))

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []


class DataLoader:
    """
    A Custom dataloader to preprocessing images and feed them to the network.

    :param minibatchlist: ([np.array]) list of observations indices (grouped per minibatch)
    :param images_path: (np.array) Array of path to images
    :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
    :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param is_training: (bool)
    :param augment: (bool) Whether to use image augmentation or not
    """

    def __init__(
        self,
        minibatchlist,
        images_path,
        n_workers=1,
        infinite_loop=True,
        max_queue_len=4,
        is_training=False,
        augment=True,
    ):
        super().__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.n_minibatches = len(minibatchlist)
        self.minibatchlist = minibatchlist
        self.images_path = images_path
        self.shuffle = is_training
        self.queue = Queue(max_queue_len)
        self.process = None
        self.augmenter = None
        if augment:
            self.augmenter = get_image_augmenter()
        self.start_process()

    @staticmethod
    def create_minibatch_list(n_samples, batch_size):
        """
        Create list of minibatches.

        :param n_samples: (int)
        :param batch_size: (int)
        :return: ([np.array])
        """
        minibatchlist = []
        for i in range(n_samples // batch_size + 1):
            start_idx = i * batch_size
            end_idx = min(n_samples, (i + 1) * batch_size)
            minibatchlist.append(np.arange(start_idx, end_idx))
        return minibatchlist

    def start_process(self):
        """Start preprocessing process"""
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend="threading") as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    indices = np.random.permutation(self.n_minibatches).astype(np.int64)
                else:
                    indices = np.arange(len(self.minibatchlist), dtype=np.int64)

                for minibatch_idx in indices:

                    images = self.images_path[self.minibatchlist[minibatch_idx]]

                    if self.n_workers <= 1:
                        batch = [self._make_batch_element(image_path, self.augmenter) for image_path in images]

                    else:
                        batch = parallel(
                            delayed(self._make_batch_element)(image_path, self.augmenter) for image_path in images
                        )

                    batch_input = np.concatenate([batch_elem[0] for batch_elem in batch], axis=0)
                    batch_target = np.concatenate([batch_elem[1] for batch_elem in batch], axis=0)

                    if self.shuffle:
                        self.queue.put((minibatch_idx, batch_input, batch_target))
                    else:
                        self.queue.put((batch_input, batch_target))

                    # Free memory
                    del batch_input, batch_target, batch

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, image_path, augmenter=None):
        """
        :param image_path: (str) path to an image
        :param augmenter: (iaa.Sequential) Image augmenter
        :return: (np.ndarray, np.ndarray)
        """
        # TODO: use mp4 video directly instead of images
        # cf https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture
        im = cv2.imread(image_path)
        if im is None:
            raise ValueError(f"tried to load {image_path}.jpg, but it was not found")

        postprocessor = CheckFliplrPostProcessor()

        if augmenter is not None:
            input_img = augmenter.augment_image(
                preprocess_image(im.copy(), normalize=False), hooks=imgaug.HooksImages(postprocessor=postprocessor)
            )
            # Normalize
            input_img = preprocess_input(input_img.astype(np.float32), mode="rl")
            input_img = input_img.reshape((1,) + input_img.shape)

        if postprocessor.flipped:
            target_img = preprocess_image(im, normalize=False)
            target_img = iaa.Fliplr(1).augment_image(target_img)
            target_img = preprocess_input(target_img.astype(np.float32), mode="rl")
            target_img = target_img.reshape((1,) + target_img.shape)
        else:
            target_img = preprocess_image(im)
            target_img = target_img.reshape((1,) + target_img.shape)
            if augmenter is None:
                input_img = target_img.copy()
        return input_img, target_img

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()
