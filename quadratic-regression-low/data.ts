import * as tf from '@tensorflow/tfjs';

interface Factor {
  a: number;
  b: number;
  c: number;
}

interface Data {
  xs: tf.Tensor;
  ys: tf.Tensor;
}

// 根据 a, b, c 生成用于训练的数据
export function generateData(num: number, factor: Factor): Data {
  const a = tf.scalar(factor.a);
  const b = tf.scalar(factor.b);
  const c = tf.scalar(factor.c);

  // 随机生成范围在 [-1, 1] 之间的 num 个数字作为 x
  const xs = tf.randomUniform([num], -1, 1)

  // 计算相应的实际 y 值
  const ys = tf.tidy(() => {
    return a.mul(xs.square())
      .add(b.mul(xs))
      .add(c);
  });

  return {
    xs,
    ys
  }
}