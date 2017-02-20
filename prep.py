'''
Utils for preprocessing
'''
import pickle
import random
import warnings
import numpy as np
import utils
from nolearn.lasagne import BatchIterator
from sklearn.utils import shuffle
from skimage import exposure
from skimage.transform import rotate, warp, ProjectiveTransform

class AugmentedSignsBatchIterator(BatchIterator):
    def __init__(self, batch_size, shuffle=False, seed=42, p=0.5, intensity=0.5):
        super(AugmentedSignsBatchIterator, self).__init__(batch_size, shuffle, seed)
        self.p = p
        self.intensity = intensity

    def transform(self, Xb, yb):
        Xb, yb = super(AugmentedSignsBatchIterator, self).transform(Xb if yb is None else Xb.copy(), yb)
        if yb is not None:
            batch_size = Xb.shape[0]
            image_size = Xb.shape[1]
            Xb = self.rotate(Xb, batch_size)
            Xb = self.apply_projection_transform(Xb, batch_size, image_size)
        return Xb, yb

    def rotate(self, Xb, batch_size):
        for i in np.random.choice(batch_size, int(batch_size*self.p), replace=False):
            delta = 30.0 * self.intensity
            Xb[i] = rotate(Xb[i], random.uniform(-delta, delta), mode='edge')
        return Xb

    def apply_projection_transform(self, Xb, batch_size, image_size):
        d = image_size * 0.3 * self.intensity
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            tl_top = random.uniform(-d, d)
            tl_left = random.uniform(-d, d)
            bl_bottom = random.uniform(-d, d)
            bl_left = random.uniform(-d, d)
            tr_top = random.uniform(-d, d)
            tr_right = random.uniform(-d, d)
            br_bottom = random.uniform(-d, d)
            br_right = random.uniform(-d, d)

            transform = ProjectiveTransform()
            transform.estimate(np.array((
                (tl_left, tl_top), (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size- br_bottom), (image_size - tr_right, tr_top)
                )), np.array((
                    (0, 0), (0, image_size),
                    (image_size, image_size), (image_size, 0)
                )))
            Xb[i] = warp(Xb[i], transform, output_shape=(image_size, image_size), order=1, mode='edge')
        return Xb

def extend_balancing_classes(X, y, aug_intensity=0.5, counts=None):
    num_classes = 43
    _, class_counts = np.unique(y, return_counts=True)
    max_c = max(class_counts)
    total = max_c * num_classes if counts is None else np.sum(counts)
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_extended = np.empty([0], dtype=y.dtype)
    print("Extending dataset using augmented data (intensity = {})".format(aug_intensity))

    for c, c_count in zip(range(num_classes), class_counts):
        max_c = max_c if counts is None else counts[c]
        X_source = (X[y == c] / 255.).astype(np.float32)
        y_source = y[y == c]
        X_extended = np.append(X_extended, X_source, axis=0)
        for i in range((max_c // c_count) - 1):
            batch_iterator = AugmentedSignsBatchIterator(batch_size=X_source.shape[0], p=1.0, intensity=aug_intensity)
            for x_batch, _ in batch_iterator(X_source, y_source):
                X_extended = np.append(X_extended, x_batch, axis=0)
                utils.print_progress(X_extended.shape[0], total)
        batch_iterator = AugmentedSignsBatchIterator(batch_size=max_c % c_count, p=1.0, intensity=aug_intensity)
        for x_batch, _ in batch_iterator(X_source, y_source):
            X_extended = np.append(X_extended, x_batch, axis=0)
            utils.print_progress(X_extended.shape[0], total)
            break
        added = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((added), c, dtype=int))
    return ((X_extended * 255.).astype(np.uint8), y_extended)

def flip_extend(X, y):
    self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    self_flippable_both = np.array([32, 40])
    cross_flippable = np.array([[19, 20], [33, 34], [36, 37], [38, 39], [20, 19], [34, 33], [37, 36], [39, 38]])
    num_classes = 43

    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_extended = np.empty([0], dtype=y.dtype)

    for c in range(num_classes):
        X_extended = np.append(X_extended, X[y == c], axis=0)
        if c in self_flippable_horizontally:
            X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis=0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        if c in cross_flippable[:, 0]:
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis=0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        if c in self_flippable_vertically:
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis=0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        if c in self_flippable_both:
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis=0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

    return (X_extended, y_extended)

def preprocess_dataset(X, y=None):
    num_classes=43
    print("Preprocessing dataset with {} examples:".format(X.shape[0]))
    X = 0.299* X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    X = (X/255.).astype(np.float32)

    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[i] = exposure.equalize_adapthist(X[i])
        utils.print_progress(i+1, X.shape[0])

    if y is not None:
        y = np.eye(num_classes)[y]
        X, y = shuffle(X, y)
    X = X.reshape(X.shape + (1,))
    return X, y

def load_pickled_data(file, columns):
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))

def load_and_process_data(pickled_data_file):
    X, y = load_pickled_data(pickled_data_file, columns=['features', 'labels'])
    X, y = preprocess_dataset(X, y)
    return (X, y)