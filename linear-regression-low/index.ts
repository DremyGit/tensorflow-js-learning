/**
 * 使用 Tensorflow.js 低阶 API 完成简单的线性回归
 *   `y = 2 * x - 1`
 * @author Dremy <dremy@dremy.cn>
 */

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

// 1. 设置需要训练的元素变量，使用随机数初始化
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));

// 2. 构建预测函数 y = a * x + b
function predict(x: tf.Tensor): tf.Tensor {
  // 在 tidy 回调中执行复杂计算，可以对内存进行优化
  return tf.tidy(() => {
    return a.mul(x)
      .add(b)
  });
}

// 3. 定义损失函数，这里使用 meanSquareError 均方差
function loss(predictions: tf.Tensor, labels: tf.Tensor): tf.Scalar {
  return predictions.sub(labels).square().mean();
}

// 4. 定义优化器
// 学习速率决定收敛的速度，太大会导致过拟合
const learningRate = 0.05
// 使用随机梯度下降作为优化器
const optimizer = tf.train.sgd(learningRate);

// 5. 定义训练循环
async function train(xs: tf.Tensor, ys: tf.Tensor, total: Number = 75) {
  for (let i = 0; i < total; i++) {
    // 通过损失函数评估参数值的准确性，多次迭代进行优化，使损失最小化
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      const lossValue = loss(predsYs, ys);
      console.log('epochs: %d, loss: %f', i + 1, lossValue.dataSync()[0]);
      return lossValue;
    })
  }
}

async function start() {
  // 使用 1, 2, 3, 4, 5 作为训练的输入样本
  const xs = tf.tensor1d([1, 2, 3, 4, 5]);

  // 使用 1, 3, 5, 7, 9 作为训练的输出样本
  const ys = tf.tensor1d([1, 3, 5, 7, 9]);

  // 对模型进行训练
  await train(xs, ys, 500);
  
  // 对输入进行预测，打印出预测结果
  const result = predict(tf.tensor1d([6]));
  result.print();
}

start();


