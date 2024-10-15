async function run() {
  const data = await getData();
  const values = data.map((m, i) => ({
    x: i + 1,
    y: parseFloat(m["growthRate"])
  }));
  tfvis.render.scatterplot(
    { name: 'Growth Rate' },
    { values },
    {
      xLabel: 'Country',
      yLabel: 'Growth Rate',
      height: 300
    }
  );
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;
  await trainModel(model, inputs, labels);
  testModel(model, data, tensorData);
}
async function getData() {
  const promise = await fetch("data.json");
  const data = await promise.json();
  const dataArr = data.map((m, i) => ({
    "country": i + 1,
    "growthRate": m["growthRate"]
  }));
  return dataArr;
}

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [1], units: 5, activation: "linear",
  }));
  model.add(tf.layers.dense({units: 15, activation: "relu"}))
  model.add(tf.layers.dense({ units: 1, }));
  return model;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);
    const inputs = data.map(m => m.country);
    const labels = data.map(m => m.growthRate);
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    const inputsMax = inputTensor.max();
    const inputsMin = inputTensor.min();
    const labelsMax = labelTensor.max();
    const labelsMin = labelTensor.min();
    const normalizedInputs = inputTensor.sub(inputsMin).div(inputsMax.sub(inputsMin));
    const normalizedLabels = labelTensor.sub(labelsMin).div(labelsMax.sub(labelsMin));
    return { inputs: normalizedInputs, labels: normalizedLabels, inputsMax, inputsMin, labelsMax, labelsMin };
  })
}

function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
    metrics: ["mse"],
  });

  return model.fit(inputs, labels, {
    batchSize: 20,
    epochs: 50,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      {
        height: 200,
        callbacks: ["onEpochEnd"]
      }
    )
  })
}

function testModel(model, inputData, normalizationData) {
  const { inputsMax, inputsMin, labelsMin, labelsMax } = normalizationData;
  const [xs, preds] = tf.tidy(() => {
    const xNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xNorm.reshape([100, 1]));
    const unNormXs = xNorm
      .mul(inputsMax.sub(inputsMin))
      .add(inputsMin);

    const unNormPreds = predictions
      .mul(labelsMax.sub(labelsMin))
      .add(labelsMin);
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });
  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  });

  const originalPoints = inputData.map(d => ({
    x: d.country, y: d.growthRate,
  }));
  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
    {
      xLabel: 'Country',
      yLabel: 'growthRate',
      height: 300
    }
  );
}

document.addEventListener('DOMContentLoaded', run);
