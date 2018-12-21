/**
 * 使用 Tensorflow.js 高阶 API 完成最简单的线性回归
 *   `y = 2 * x - 1`
 * @author Dremy <dremy@dremy.cn>
 */

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

async function start() {
  // 创建一个模型
  const model = tf.sequential();

  // 添加输入层，这里也作为输出层，使用一个输出神经元，输入形状为[*, 1]
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // 编译模型
  //   损失函数使用均方差
  //   优化器使用随机梯度下降（Stochastic Gradient Descent）
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // 使用 1, 2, 3, 4, 5 作为训练的输入样本
  const xs = tf.tensor1d([1, 2, 3, 4, 5]);

  // 使用 1, 3, 5, 7, 9 作为训练的输出样本
  const ys = tf.tensor1d([1, 3, 5, 7, 9]);

  // 对模型进行训练，训练 500 次
  await model.fit(xs, ys, { epochs: 500 });

  // 对输入进行预测，并打印出预测结果
  const result = model.predict(tf.tensor1d([6])) as tf.Tensor;
  result.print();
}

start();
