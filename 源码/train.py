import tensorflow as tf

from process_data import train_ds, test_ds
from models import build_model

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10,
                                              restore_best_weights=True)]

model.fit(train_ds,
          validation_data=test_ds,
          epochs=100,
          callbacks=callbacks,
          verbose=2)

model.save("../模型/results(2)")
