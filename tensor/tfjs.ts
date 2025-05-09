import * as tf from "@tensorflow/tfjs";

const tfjs = async () => {
  // 예시 데이터 (몸무게, 스내치 1RM → Isabel 기록)
  const 원인 = tf.tensor2d([
    [70, 155],
    [75, 155],
    [80, 195],
    [85, 195],
    [90, 205],
  ]);

  const 결과 = tf.tensor2d([
    [280], // Isabel 시간(초)
    [180],
    [150],
    [140],
    [130],
  ]);

  // 모델 구성
  const X = tf.input({ shape: [2] });
  const dense = tf.layers.dense({ units: 1 }).apply(X) as tf.SymbolicTensor;
  const model = tf.model({ inputs: X, outputs: dense });

  // 모델 컴파일 & 학습
  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.sgd(0.01),
  });

  await model.fit(원인, 결과, {
    epochs: 500,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 50 === 0)
          console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      },
    },
  });

  // 예측 예시 (입력 값도 정규화 필요)
  const 예측입력 = tf.tensor2d([[85, 225]]);
  const 예측 = model.predict(예측입력);
  예측.print(); // 예상 Isabel 기록 출력
};

export default tfjs;
