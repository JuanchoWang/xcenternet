from random import shuffle
import tensorflow as tf

from xcenternet.datasets.dataset import Dataset


class McodDataset(Dataset):
    def __init__(self, dataset_path_tr, dataset_path_te, init_lr):
        self.features = {
            TfExampleFields.height: tf.io.FixedLenFeature((), dtype=tf.int64, default_value=1),
            TfExampleFields.width: tf.io.FixedLenFeature((), dtype=tf.int64, default_value=1),
            TfExampleFields.colorspace: tf.io.FixedLenFeature((), dtype=tf.string, default_value=''),
            TfExampleFields.channels: tf.io.FixedLenFeature((), dtype=tf.int64, default_value=1),
            TfExampleFields.image_class_label: tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            TfExampleFields.image_class_synset: tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            TfExampleFields.image_class_text: tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
            TfExampleFields.object_bbox_ymin: tf.io.VarLenFeature(tf.float32),
            TfExampleFields.object_bbox_xmin: tf.io.VarLenFeature(tf.float32),
            TfExampleFields.object_bbox_ymax: tf.io.VarLenFeature(tf.float32),
            TfExampleFields.object_bbox_xmax: tf.io.VarLenFeature(tf.float32),
            TfExampleFields.image_format: tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            TfExampleFields.filename: tf.io.FixedLenFeature((), tf.string, default_value=''),
            TfExampleFields.image_encoded: tf.io.FixedLenFeature((), tf.string, default_value=''),
        }
        num_classes = 9
        self.path_train_set = dataset_path_tr
        self.path_val_set = dataset_path_te
        super().__init__(num_classes, init_lr)

    def scheduler(self, epoch):
        if epoch < 40:
            return self.initial_learning_rate
        elif epoch < 80:
            return self.initial_learning_rate * 0.1
        else:
            return self.initial_learning_rate * 0.01

    def decode(self, data):
        """Return a single image and associated label and bounding box
        Copied from Modellbau

        Args:
          data: a string tensor holding a serialized protocol buffer corresponding
            to data for a single image

        Returns:
          image: image tensor
          label: class label tensor
          bbox: bounding box comprising the annotated object
        """
        par_feat = tf.io.parse_single_example(data, self.features)

        image = tf.io.decode_image(par_feat[TfExampleFields.image_encoded], channels=3)
        image.set_shape([None, None, 3])

        object_bbox_xmin = tf.sparse.to_dense(par_feat[TfExampleFields.object_bbox_xmin])
        object_bbox_xmax = tf.sparse.to_dense(par_feat[TfExampleFields.object_bbox_xmax])
        object_bbox_ymin = tf.sparse.to_dense(par_feat[TfExampleFields.object_bbox_ymin])
        object_bbox_ymax = tf.sparse.to_dense(par_feat[TfExampleFields.object_bbox_ymax])
        bbox = tf.stack([object_bbox_ymin, object_bbox_xmin, object_bbox_ymax, object_bbox_xmax], axis=-1)
        label = par_feat[TfExampleFields.image_class_label]

        # added by Xiao
        image_id = par_feat[TfExampleFields.filename]
        labels = tf.reshape(label, [-1])
        bboxes = tf.reshape(bbox, [-1, 4])

        return image, labels, bboxes, image_id

    def _load_dataset(self, filenames, shuffle_tfrecords=True):
        if shuffle_tfrecords:
            shuffle(filenames)
        ds = tf.data.TFRecordDataset(filenames)
        return ds

    def load_train_datasets(self):
        dataset_train = self._load_dataset(filenames=self.path_train_set)
        dataset_train_size = sum(1 for _ in dataset_train)
        return dataset_train, dataset_train_size

    def load_validation_datasets(self):
        dataset_valid = self._load_dataset(filenames=self.path_val_set)
        dataset_valid_size = sum(1 for _ in dataset_valid)
        return dataset_valid, dataset_valid_size


class TfExampleFields(object):
    """TF-example proto feature names

    Holds the standard feature names to load from an Example proto

    Attributes:
      image_encoded: image encoded as string
      image_format: image format, e.g. "JPEG"
      filename: filename
      channels: number of channels of image
      colorspace: colorspace, e.g. "RGB"
      height: height of image in pixels, e.g. 462
      width: width of image in pixels, e.g. 581
      source_id: original source of the image
      image_class_text: image-level label in text format
      image_class_label: image-level label in numerical format
      object_class_text: labels in text format, e.g. ["person", "cat"]
      object_class_label: labels in numbers, e.g. [16, 8]
      object_bbox_xmin: xmin coordinates of groundtruth box, e.g. 10, 30
      object_bbox_xmax: xmax coordinates of groundtruth box, e.g. 50, 40
      object_bbox_ymin: ymin coordinates of groundtruth box, e.g. 40, 50
      object_bbox_ymax: ymax coordinates of groundtruth box, e.g. 80, 70
      ignore_area_bbox_xmin: xmin coordinates of ignore box, e.g. 10, 30
      ignore_area_bbox_xmax: xmax coordinates of ignore box, e.g. 50, 40
      ignore_area_bbox_ymin: ymin coordinates of ignore box, e.g. 40, 50
      ignore_area_bbox_ymax: ymax coordinates of ignore box, e.g. 80, 70
      object_view: viewpoint of object, e.g. ["frontal", "left"]
      object_truncated: is object truncated, e.g. [true, false]
      object_occluded: is object occluded, e.g. [true, false]
      object_difficult: is object difficult, e.g. [true, false]
      object_group_of: is object a single object or a group of objects
      object_depiction: is object a depiction
      object_segment_area: the area of the segment.
      object_weight: a weight factor for the object's bounding box.
      instance_masks: instance segmentation masks.
      instance_boundaries: instance boundaries.
      instance_classes: Classes for each instance segmentation mask.
      detection_class_label: class label in numbers.
      detection_bbox_ymin: ymin coordinates of a detection box.
      detection_bbox_xmin: xmin coordinates of a detection box.
      detection_bbox_ymax: ymax coordinates of a detection box.
      detection_bbox_xmax: xmax coordinates of a detection box.
      detection_score: detection score for the class label and box.
    """
    image_encoded = 'image/encoded'
    image_format = 'image/format'  # format is reserved keyword
    key = 'image/key/sha256'
    filename = 'image/filename'
    channels = 'image/channels'
    colorspace = 'image/colorspace'
    height = 'image/height'
    width = 'image/width'
    source_id = 'image/source_id'
    image_class_text = 'image/class/text'
    image_class_label = 'image/class/label'
    image_class_synset = 'image/class/synset'
    object_class_text = 'image/object/class/text'
    object_class_label = 'image/object/class/label'
    ignore_area_bbox_xmin = 'image/ignore_area/bbox/xmin'
    ignore_area_bbox_ymin = 'image/ignore_area/bbox/ymin'
    ignore_area_bbox_xmax = 'image/ignore_area/bbox/xmax'
    ignore_area_bbox_ymax = 'image/ignore_area/bbox/ymax'
    object_bbox_ymin = 'image/object/bbox/ymin'
    object_bbox_xmin = 'image/object/bbox/xmin'
    object_bbox_ymax = 'image/object/bbox/ymax'
    object_bbox_xmax = 'image/object/bbox/xmax'
    object_bbox_label = 'image/object/bbox/label'
    object_bbox_split_line = 'image/object/bbox/vl'
    object_bbox_split_type = 'image/object/bbox/vertical_line_type'
    object_view = 'image/object/view'
    object_truncated = 'image/object/truncated'
    object_occluded = 'image/object/occluded'
    object_difficult = 'image/object/difficult'
    object_group_of = 'image/object/group_of'
    object_depiction = 'image/object/depiction'
    object_segment_area = 'image/object/segment/area'
    object_weight = 'image/object/weight'
    semseg_channels = 'image/segmentation/channels'
    semseg_data = 'image/segmentation/data'
    semseg_format = 'image/segmentation/format'
    semseg_height = 'image/segmentation/height'
    semseg_width = 'image/segmentation/width'
