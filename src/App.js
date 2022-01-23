import React, { useState } from "react";
import { Button, Container, Form, Row, Col } from "react-bootstrap";
import { Scatter } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";
//import { seAnnModel } from "./ann/models";
import "./App.css";
import * as tf from "@tensorflow/tfjs";

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

function App() {
  const [funStr, setFunStr] = useState("Math.sin(x)");
  const [dataSize, setDataSize] = useState(200);
  const [minx, setMinX] = useState(-10);
  const [maxx, setMaxX] = useState(10);
  const [varx, setVarX] = useState(0.1);
  const [hln, setHiddenLayer] = useState(3);
  const [nun, setNeuron] = useState(32);
  const [epochs, setEpochs]= useState(10)
  const [batchsize, setBatchSize]= useState(32)
  const [activation, setActivation] = useState("relu");
  const [optimizer, setOptimizer] = useState("adam");
  const [loss, setLoss] = useState("meanSquaredError");
  const [Data, setData] = useState([]);
  const [chartdata, setChartData] = useState(null);
  const [chartoption, setChartOption] = useState({
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  });
  // var Data = []
  var annModel;

  const createModel = (layers, neurons, activation, loss, opt) => {
    var model = tf.sequential();
    model.add(tf.layers.dense({ units: neurons, inputShape: [1] }));

    for (var i = 0; i < layers; i++) {
      model.add(
        tf.layers.dense({
          units: neurons,
          inputShape: [neurons],
          activation: activation,
        })
      );
    }
    model.add(tf.layers.dense({ units: 1, inputShape: [neurons] }));

    model.compile({
      loss: loss,
      optimizer: opt,
      metrics: [tf.metrics.meanSquaredError],
    });

    return model;
  };

  const fitNN = async (loop) => {
    let X = Data.map((item) => item.x);
    let Y = Data.map((item) => item.y);

    let predict_x = [...Array(50).keys()].map(
      (x) => x * (1 / 50) * (maxx - minx) + minx
    );
    await annModel.fit(tf.tensor(X), tf.tensor(Y), {
      epochs: epochs,
      batchSize: parseInt(batchsize),
      shuffle: true,
    });

    let prediction = await annModel
      .predict(tf.tensor2d(predict_x, [predict_x.length, 1]))
      .dataSync();

    let plot_data = [];
    for (var i = 0; i < predict_x.length; i++) {
      plot_data.push({ x: predict_x[i], y: prediction[i] });
    }

    const chd = { ...chartdata };
    if (chartdata.datasets.length > 1) {
      chd.datasets.pop();
    }
    chd.datasets = [
      chd.datasets[0],
      {
        label: `Fit (Iteration: ${loop})`,
        borderColor: "rgb(0, 0, 255)",
        data: plot_data,
        showLine: true,
        pointRadius: 0,
      },
    ];
    setChartData(chd);

    if(loop>0){
      fitNN(loop-1);
    }    
  };

  const fitAnnModel = async () => {
    if (Data.length > 1) {
      annModel = createModel(parseInt(hln), parseInt(nun), activation, loss, optimizer);
      fitNN(epochs);
    } else {
      alert("Befor to Fit Model, First Generate Data");
    }
  };

  const plotData = () => {
    generateData(parseInt(dataSize), parseFloat(minx), parseFloat(maxx), parseFloat(varx), funStr).then((gendata) => {
      setData(gendata);
      const chartd = {
        datasets: [
          {
            label: "Generated Data",
            data: gendata,
            backgroundColor: "rgba(255, 99, 132, 1)",
          },
        ],
      };
      setChartData(chartd);
    });
  };

  const generateData = async (n, minx, maxx, varx, funStr) => {
    let data = [];
    const func = (x) => {
      return eval(funStr);
    };
    for (var i = 0; i < n; i++) {
      let x = Math.random() * (maxx - minx) + minx;
      let y = func(x) + (Math.random() * 2 * varx - varx);
      data.push({ x: x, y: y });
    }
    return data;
  };


  return (
    <React.Fragment>
      <Container>
        <br />
        <h3>
          This code demonstrate how deep learning works through the Global
          Function Approximation using Tensorflow.js
        </h3>
        <br />
        <Row>
          <Form.Group>
            <Form.Label>Enter Function to Fit (JavaScript Format) </Form.Label>
            <Form.Control
              type="text"
              defaultValue={funStr}
              placeholder="Enter Function to Fit"
              onChange={(e) => setFunStr(e.target.value)}
            />
          </Form.Group>
        </Row>
        <br />
        <Row>
          <Col>
            <Form.Group>
              <Form.Label>Enter Data Size</Form.Label>
              <Form.Control
                type="text"
                defaultValue={dataSize}
                placeholder="Enter Data Size"
                onChange={(e) => setDataSize(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Enter min X</Form.Label>
              <Form.Control
                type="text"
                defaultValue={minx}
                placeholder="Enter min X"
                onChange={(e) => setMinX(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Enter max X</Form.Label>
              <Form.Control
                type="text"
                defaultValue={maxx}
                placeholder="Enter max X"
                onChange={(e) => setMaxX(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Enter variance X</Form.Label>
              <Form.Control
                type="text"
                defaultValue={varx}
                placeholder="Enter variance X"
                onChange={(e) => setVarX(e.target.value)}
              />
            </Form.Group>
          </Col>
        </Row>
        <br />
        <Row>
          <Col>
            <Form.Group>
              <Form.Label>Select Activation</Form.Label>
              <Form.Select onChange={(e) => setActivation(e.target.value)}>
                <option vllue="relu">Rectified Linear Unit</option>
                <option vllue="elu">Exponential Linear Unit</option>
                <option value="softmax">Softmax</option>
                <option value="sigmoid">Sigmoid</option>
                <option value="linear">Linear</option>
              </Form.Select>
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Select Optimizer</Form.Label>
              <Form.Select onChange={(e) => setOptimizer(e.target.value)}>
                <option vllue="adam">Adam</option>
                <option value="adamax">Adamax</option>
                <option vllue="sgd">Stochastic Gradient Descent</option>
                <option value="rmsprop">RMSProp</option>
              </Form.Select>
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Select Loss</Form.Label>
              <Form.Select onChange={(e) => setLoss(e.target.value)}>
                <option value="meanSquaredError">Mean Squared Error</option>
                <option value="computeWeightedLoss">
                  Compute Weighted Loss
                </option>
                <option vllue="cosineDistance">Cosine Difference</option>
                <option value="hingeLoss">Hinge Loss</option>
                <option value="huberLoss">Huber Loss</option>
                <option value="logLoss">Log Loss</option>
                <option value="meanSquaredError">Mean Squared Error</option>
                <option value="sigmoidCrossEntropy">
                  Sigmoid Cross Entropy
                </option>
                <option value="softmaxCrossEntropy">
                  Softmax Cross Entropy
                </option>
                <option vllue="absoluteDifference">Absolute Difference</option>
              </Form.Select>
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Hidden Layers</Form.Label>
              <Form.Control
                type="text"
                defaultValue={hln}
                placeholder="Number of Hidden Layers"
                onChange={(e) => setHiddenLayer(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Neurons</Form.Label>
              <Form.Control
                type="text"
                defaultValue={nun}
                placeholder="Number Neuron Per Layer"
                onChange={(e) => setNeuron(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Epochs</Form.Label>
              <Form.Control
                type="text"
                defaultValue={epochs}
                placeholder="Epochs"
                onChange={(e) => setEpochs(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Batch Size</Form.Label>
              <Form.Control
                type="text"
                defaultValue={batchsize}
                placeholder="Batch Size"
                onChange={(e) => setBatchSize(e.target.value)}
              />
            </Form.Group>
          </Col>
        </Row>
        <br />
        <Button variant="primary" onClick={plotData}>
          Generate Data
        </Button>
        &nbsp;&nbsp;
        <Button variant="warning" onClick={fitAnnModel}>
          Fit Sequntial Model To Genereted Data
        </Button>
        <Row>
          {chartdata ? <Scatter options={chartoption} data={chartdata} /> : ""}
        </Row>
      </Container>
    </React.Fragment>
  );
}

export default App;
