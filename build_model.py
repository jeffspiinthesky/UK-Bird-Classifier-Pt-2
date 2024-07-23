import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data = object_detector.DataLoader.from_pascal_voc(
    'label_birds/birds_model/train',
    'label_birds/birds_model/train',
    ['Wood_Pigeon', 'House_Sparrow', 'Robin', 'Magpie', 'White_Collared_Dove', 'Starling']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'label_birds/birds_model/validate',
    'label_birds/birds_model/validate',
    ['Wood_Pigeon', 'House_Sparrow', 'Robin', 'Magpie', 'White_Collared_Dove', 'Starling']
)
spec = model_spec.get('efficientdet_lite0')
print('Creating the data model')
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=75, validation_data=val_data)
print('Validating the data model pre-export')
results = model.evaluate(val_data)
print(f'Results from pre-exported model: {results}')
print('Exporting the data model')
model.export(export_dir='.', tflite_filename='bird_model.tflite')
print('Validating the exported data model')
results = model.evaluate_tflite('bird_model.tflite', val_data)
print(f'Results from exported model: {results}')

