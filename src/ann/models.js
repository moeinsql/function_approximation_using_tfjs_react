import * as tf from "@tensorflow/tfjs";
//import { model } from "@tensorflow/tfjs";

const oneHot = (val, categoryCount) =>
  Array.from(tf.oneHot(val, categoryCount).dataSync());

export const perceptron = ({ x, w, bias }) => {
  const product = tf.dot(x, w).dataSync()[0];
  return product + bias < 0 ? 0 : 1;
};

export const sigmoidPerceptron = ({ x, w, bias }) => {
  const product = tf.dot(x, w).dataSync()[0];
  return tf.sigmoid(product + bias).dataSync()[0];
};

export async function seAnnModel(xval, yval) {
  const X = tf.tensor2d([
    // pink, small
    [0.1, 0.1],
    [0.3, 0.3],
    [0.5, 0.6],
    [0.4, 0.8],
    [0.9, 0.1],
    [0.75, 0.4],
    [0.75, 0.9],
    [0.6, 0.9],
    [0.6, 0.75],
  ]);
  // 0 - no buy, 1 - buy
  const y = tf.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1].map((i) => oneHot(i, 2)));
  console.log(y.dataSync())

  const seModel = tf.sequential();
  seModel.add(
    tf.layers.dense({
      inputShape: [2],
      units: 3,
      activation: "relu",
    })
  );

  seModel.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
    })
  );

  seModel.compile({
    optimizer: tf.train.adam(0.1),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  await seModel.fit(X, y, {
    shuffle: true,
    epochs: 20,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch " + epoch);
        console.log("Loss: " + logs.loss + " accuracy: " + logs.acc);
      },
    },
  });

  return seModel.predict(tf.tensor2d([[xval, yval]])).dataSync();
}
