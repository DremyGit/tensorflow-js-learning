/**
 * 使用 Tensorflow.js 低阶 API 完成逻辑回归
 * ```
 *  y = 0 (x < 0)
 *  y = 1 (x > 0) 
 * ```
 * @author Dremy <dremy@dremy.cn>
 */

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

// 维度/特征 数量
const FEATURES = 1
// 分类/标签 总数
const LABELS = 2;

// 权重 Weight
const W = tf.variable(tf.zeros([FEATURES, LABELS]));
// 偏置 bias
const b = tf.variable(tf.zeros([LABELS]));

/**
 * 使用 sigmoid 函数作为预测函数
 * http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
 * @param x 特征值输入
 * @return 每种分类的概率
 */
function predict(x: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    // 使用 softmax 进行预测
    return x.matMul(W).add(b).softmax();
  });
}

/**
 * 使用交叉熵作为损失函数
 * @param predictions 预测值
 * @param labels 实际标签值
 * @returns 交叉熵
 */
function loss(predictions: tf.Tensor, labels: tf.Tensor1D): tf.Scalar {
  // 将 label 转为每种分类的可能性
  //   0 => [1, 0]
  //   1 => [0, 1]
  const y = tf.oneHot(labels, LABELS);

  // 求交叉熵 cross entropy
  //  avg(-sum(log(pred) / y))
  return predictions.log().mul(y).sum(1).neg().mean() as tf.Scalar;
}

// 使用随机梯度下降进行优化
const learningRate = 0.02
const optimizer = tf.train.sgd(learningRate);

async function train(xs: tf.Tensor, labels: tf.Tensor1D, total: Number) {
  for (let i = 0; i < total; i++) {
    // 通过损失函数评估参数值的准确性，多次迭代进行优化，使损失最小化
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      const lossValue = loss(predsYs, labels);
      console.log('epochs: %d, loss: %f', i + 1, lossValue.dataSync()[0]);
      return lossValue;
    })
  }
}

async function start() {
  // 定义用于训练的输入输出数据
  const xs = tf.tensor2d([-30, -20, -10, 10, 20, 30], [6, 1]);
  const ys = tf.tensor1d([0, 0, 0, 1, 1, 1], 'int32');

  // 对模型进行训练
  await train(xs, ys, 1000);
  
  // 输出预测结果
  const result = predict(tf.tensor2d([-2, 0, 1], [3, 1])) as tf.Tensor;
  result.print(); // 每种分类的可能性
  result.argMax(1).print(); // 预测输出值

}

start();
