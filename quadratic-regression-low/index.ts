/**
 * 使用 Tensorflow.js 低阶 API 完成二次回归
 *   `y = a * x^2 + b * x + c`
 * @author Dremy <dremy@dremy.cn>
 */

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import { generateData } from './data';

// 1. 设置需要训练的元素变量，使用随机数初始化
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));

// 2. 构建预测函数 y = a * x^2 + b * x + c
function predict(x: tf.Tensor): tf.Tensor {
  // 在 tidy 回调中执行复杂计算，可以对内存进行优化
  return tf.tidy(() => {
    return a.mul(x.square())
      .add(b.mul(x))
      .add(c)
  });
}

// 3. 定义损失函数，这里使用 meanSquareError 均方差
function loss(predictions: tf.Tensor, labels: tf.Tensor): tf.Scalar {
  return predictions.sub(labels).square().mean();
}

// 4. 定义优化器
// 学习速率决定收敛的速度，太大会导致过拟合
const learningRate = 0.2
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
  // 为 a, b, c 随机取值
  const ra = Math.random();
  const rb = Math.random();
  const rc = Math.random();

  // 随机生成用于训练的输入输出数据
  const { xs, ys } = generateData(10, { a: ra, b: rb, c: rc });

  // 对模型进行训练
  await train(xs, ys, 500);
  
  // 分别输出 a, b, c 的实际值和预测值
  console.log({
    a: {
      real: ra,
      pred: a.dataSync()[0]
    },
    b: {
      real: rb,
      pred: b.dataSync()[0]
    },
    c: {
      real: rc,
      pred: c.dataSync()[0]
    }
  });
}

start();


