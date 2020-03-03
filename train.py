# docker run -it --rm --name shaoyu_clippy -u $(id -u):$(id -g) -v $PWD:/tmp -w /tmp tensorflow/tensorflow:2.0.0-gpu-py3 bash
# docker exec -it shaoyu_clippy bash
import json
from absl import flags
from absl import logging
from absl import app
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from os import path
import random

FLAGS = flags.FLAGS


if __name__ == "__main__":
    flags.DEFINE_string("params", "{}", "hyperparameters")


def normalize(img):
    img = tf.cast(img, dtype=tf.float32)
    img = (img / 127.5) - 1
    return img


def denormalize(img):
    img = img + 1
    img = img * 127.5
    return tf.cast(img, dtype=tf.uint8)


def new_model():
    inputs = tf.keras.layers.Input(shape=(240, 320, 3), dtype=tf.uint8)
    resized = tf.image.resize(inputs, [60, 80])
    x = normalize(inputs)

    x = tf.keras.layers.Conv2D(32, 4, strides=2, padding="same")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = tf.nn.relu(x)

    x = tf.math.reduce_max(x, axis=[1, 2])
    x = tf.reshape(x, [-1, 256])

    x = tf.keras.layers.Dense(2)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


class Data():
    def __init__(self, folder):
        self._trues = [path.join(folder, "true", f) for f in os.listdir(path.join(folder, "true")) if path.isfile(path.join(folder, "true", f))]
        self._falses = [path.join(folder, "false", f) for f in os.listdir(path.join(folder, "false")) if path.isfile(path.join(folder, "false", f))]
        logging.info("trues: %d, falses: %d", len(self._trues), len(self._falses))

    def get(self, batch_size):
        x = np.zeros([batch_size, 240, 320, 3], dtype=np.uint8)
        y = np.zeros([batch_size], dtype=np.int32)
        for b in range(batch_size):
            y[b] = random.randint(0, 1)
            if y[b] == 0:
                series = self._falses
            else:
                series = self._trues
            idx = random.randint(0, len(series)-1)
            x[b] = np.array(Image.open(series[idx]))
        res = {}
        res["x"] = x
        res["y"] = y
        return res


def get_params():
    params = {}
    
    root = "/home/shaoyu/repo/clippy"
    root = "/tmp"
    params["dir"] = root+"/experiments/test1"
    params["train_dir"] = root+"/data/clippyImg/train"
    params["test_dir"] = root+"/data/clippyImgVal/validation"
    params["learning_rate"] = 1e-3

    return params


def main(argv):
    params = json.loads(flags.FLAGS.params)
    if len(params) == 0:
        params = get_params()
    logging.info("PARAMS %s", json.dumps(params))

    train_data = Data(params["train_dir"])
    test_data = Data(params["test_dir"])
    model = new_model()
    optimizer = tf.keras.optimizers.RMSprop(params["learning_rate"])

    tracked = {}
    tracked["step"] = tf.Variable(1)
    tracked["model"] = model
    tracked["optimizer"] = optimizer

    ckpt = tf.train.Checkpoint(**tracked)
    manager = tf.train.CheckpointManager(ckpt, params["dir"]+"/ckpt", max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    def train(batch):
        with tf.GradientTape() as tape:
            logits = model(batch["x"])
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch["y"], logits=logits)
            loss = tf.reduce_mean(ce)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    def evaluate(data):
        total = 0
        corrects = 0
        loss_sum = 0
        for _ in range(10):
            batch = data.get(32)
            logits = model(batch["x"])
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(batch["y"], logits)
            loss = tf.reduce_sum(ce)
            pred = tf.math.argmax(logits, axis=-1)
            
            total += batch["y"].shape[0]
            corrects += tf.math.count_nonzero(pred == batch["y"])
            loss_sum += loss
        res = {}
        res["accuracy"] = float(corrects) / total
        res["loss"] = loss_sum / total
        return res

    def evaluate_all(data):
        def eval_b(batch, n):
            logits = model(batch["x"])
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(batch["y"], logits)
            pred = tf.math.argmax(logits, axis=-1)

            corrects = tf.math.count_nonzero(pred[:n] == batch["y"][:n])
            loss_sum = tf.reduce_sum(ce[:n])
            return corrects, loss_sum

        def run(files, label):
            batch_size = 32
            total = 0
            corrects = 0
            loss_sum = 0

            b = 0
            batch = {}
            batch["x"] = np.zeros([batch_size, 240, 320, 3], dtype=np.uint8)
            batch["y"] = np.zeros([batch_size], dtype=np.int32)
            for _, f in enumerate(files):
                batch["x"][b] = np.array(Image.open(f))
                batch["y"][b] = label
                b += 1

                if b >= batch_size:
                    batch_corrects, batch_loss = eval_b(batch, b)
                    total += b
                    corrects += batch_corrects
                    loss_sum += batch_loss
                    b = 0

            batch_corrects, batch_loss = eval_b(batch, b)
            total += b
            corrects += batch_corrects
            loss_sum += batch_loss

            return total, corrects, loss_sum

        total = 0
        corrects = 0
        loss = 0
        true_total, true_corrects, true_loss = run(data._trues, 1)
        total += true_total
        corrects += true_corrects
        loss += true_loss
        false_total, false_corrects, false_loss = run(data._falses, 0)
        total += false_total
        corrects += false_corrects
        loss += false_loss
        
        res = {}
        res["accuracy"] = float(corrects) / total
        res["loss"] = loss / total
        return res

    for _ in range(9999999):
        train(train_data.get(32))

        ckpt.step.assign_add(1)
        step = int(ckpt.step)
        if step < 2 or step % 1 == 0:
            save_path = manager.save()
            logging.info("Saved checkpoint for step %s: %s", step, save_path)

            train_eval = evaluate(train_data)
            test_eval = evaluate_all(test_data)

            val = {}
            val["step"] = step
            for k, v in train_eval.items():
                val["train_"+k] = float(v)
            for k, v in test_eval.items():
                val["test_"+k] = float(v)
            logging.info("LOG %s", json.dumps(val))


if __name__ == "__main__":
    app.run(main)
