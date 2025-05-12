import * as tf from "@tensorflow/tfjs";

const tfjs = async () => {
  // 예시 데이터 (몸무게, 스내치 1RM, 경력(개월) → Isabel 기록)
  const 원인 = tf.tensor2d([
    [64, 135, 6],
    [65, 135, 5],
    [66, 135, 14],
    [67, 135, 25],
    [68, 135, 8],
    [69, 145, 10],
    [70, 145, 9],
    [71, 155, 12],
    [75, 155, 24],
    [80, 195, 22],
    [85, 195, 16],
    [90, 205, 26],
  ]);

  const 결과 = tf.tensor2d([
    // Isabel 시간(초)
    [600],
    [590],
    [590],
    [550],
    [550],
    [540],
    [560],
    [280],
    [180],
    [150],
    [140],
    [130],
  ]);

  // 데이터 정규화: (각 값 - 최소값) / (최대값 - 최소값)
  const 원인_최소 = tf.min(원인, 0);
  const 원인_최대 = tf.max(원인, 0);
  const 정규화된_원인 = 원인.sub(원인_최소).div(원인_최대.sub(원인_최소));

  // 모델 구성
  const X = tf.input({ shape: [3] });
  const H = tf.layers
    .dense({ units: 8, activation: "relu" })
    .apply(X) as tf.SymbolicTensor;
  const output = tf.layers.dense({ units: 1 }).apply(H) as tf.SymbolicTensor;
  const model = tf.model({ inputs: X, outputs: output });

  // 모델 컴파일 & 학습
  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(0.01),
  });

  await model.fit(정규화된_원인, 결과, {
    epochs: 500,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 50 === 0)
          console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      },
    },
  });

  // 예측 예시 (입력 값도 정규화 필요)
  const 예측입력 = tf.tensor2d([[85, 225, 25]]);
  const 예측_정규화 = 예측입력.sub(원인_최소).div(원인_최대.sub(원인_최소));
  const 예측 = model.predict(예측_정규화);
  예측.print(); // 예상 Isabel 기록 출력
};

export default tfjs;
