/**
 * 使用 Tensorflow.js 高阶 API 完成逻辑回归
 * ```
 *  y = 0 (x < 0)
 *  y = 1 (x > 0) 
 * ```
 * @author Dremy <dremy@dremy.cn>
 */

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

// 维度/特征 数量
const FEATURES = 1;

// 分类/标签 总数
const LABELS = 2;

async function start() {
  // 创建一个模型
  const model = tf.sequential();

  // 添加输入层，使用一个输出神经元，输入形状为[*, DEMESION]
  model.add(tf.layers.dense({ units: 1, inputShape: [FEATURES] }));
  
  // 添加输出层，使用数量为 TOTAL 个输出神经元，并且使用 softmax 作为激活函数
  model.add(tf.layers.dense({ units: LABELS, activation: 'softmax' }));

  // 编译模型
  //   损失函数使用分类交叉熵
  //   优化器使用随机梯度下降
  //   使用准确率作为度量
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'sgd',
    metrics: ['accuracy']
  });

  // 定义训练样本的特征(x)和分类(y)
  const xs = tf.tensor2d([-30, -20, -10, 10, 20, 30], [6, 1]);
  const ys = tf.tensor1d([0, 0, 0, 1, 1, 1], 'int32');

  // 需要将分类标签值转为可能性矩阵后进行训练
  await model.fit(xs, tf.oneHot(ys, 2), { epochs: 500 });

  // 输出预测结果
  const result = model.predict(tf.tensor1d([-2, 0, 1])) as tf.Tensor;
  result.print(); // 每种分类的可能性
  result.argMax(1).print(); // 预测输出值
}

start();
