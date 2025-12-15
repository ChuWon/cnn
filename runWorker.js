const file = 'mnist_train.csv';

let cpuCount;
if (typeof window === 'undefined') {
	globalThis.Worker = require('worker_threads').Worker;
	cpuCount = require('os').cpus().length;

	require('fs/promises').readFile(file, { encoding: 'utf8' }).then(parse);
} else {
	cpuCount = navigator.hardwareConcurrency;
	fetch(file).then(res => res.text()).then(parse);
}

const workerCount = Math.max(1, cpuCount - 2);
console.log(`worker count: ${workerCount}`);

const workers = [];

for (let i = 0; i < workerCount; i++) {
	const worker = new Worker('./worker.js');
	worker.onmessage = onMessageWeb;
	worker.on && worker.on('message', onMessage);
	workers.push(worker);
}

function onMessageWeb(event) {
	onMessage(event.data);
}

function onMessage(msg) {
	switch (msg.id) {
		case 'paramsUpdated':
			if (--pendingParamUpdates <= 0) {
				paramUpdate.resolve();
			}
			break;

		case 'executed':
			tasks[msg.taskId].resolve(msg.result);
			break;

		default:
			console.log(`unknown msg from worker: ${msg.id}`);
	}
}

let paramUpdate;
let pendingParamUpdates = 0;

function updateParams(params, grad) {
	pendingParamUpdates++;

	const worker = workers[Math.floor(Math.random() * workers.length)];
	worker.postMessage({
		id: 'updateParams', 
		params, 
		grad, 
		learningRate
	});

	paramUpdate = {}
	paramUpdate.promise = new Promise(resolve => {
		paramUpdate.resolve = resolve;
	});
}

const tasks = {};

let workerIndex = 0;
function execute(name, args) {
	const taskId = Math.floor(Math.random() * 4666_420_69969);

	const task = {};
	task.promise = new Promise((resolve, reject) => {
		task.resolve = resolve;
		task.reject = reject;
	});
	tasks[taskId] = task;

	const worker = workers[workerIndex];
	workerIndex = (workerIndex + 1) % workers.length;

	worker.postMessage({
		id: 'execute', 
		taskId, 
		name, 
		args
	});

	return task.promise;
}

class Conv {
	constructor(inputSize, inputDepth, kernelSize, depth) {
		this.inputSize = inputSize;
		this.inputDepth = inputDepth;
		this.kernelSize = kernelSize;
		this.depth = depth;

		this.outputSize = inputSize - kernelSize + 1;
		this.inputLength = inputDepth * inputSize * inputSize;
		this.outputLength = this.depth * this.inputDepth * this.outputSize * this.outputSize;

		this.kernels = createParams(this.depth * this.inputDepth * this.kernelSize * this.kernelSize, 0.01);
		this.biases = createParams(this.outputLength, 0.01);
	}

	async forward(x) {
		this.x = x;
		return await execute('correlate', [this.inputSize, this.inputDepth, this.kernelSize, this.depth, x, this.kernels, this.biases]);
	}

	async backward(grad) {
		let kernelGrad = execute('correlate', [this.inputSize, this.inputDepth, this.outputSize, this.depth, this.x, grad]);
		let inputGrad = execute('convolve', [this.outputSize, this.inputDepth, this.kernelSize, this.depth, grad, this.kernels]);
		let biasGrad = execute('sum', [grad, this.outputLength]);

		kernelGrad = await kernelGrad;
		inputGrad = await inputGrad;
		biasGrad = await biasGrad;

		updateParams(this.kernels, kernelGrad);
		updateParams(this.biases, biasGrad);

		return inputGrad;
	}
}

class Activation {
	constructor(f, fPrime) {
		this.f = f;
		this.fPrime = fPrime;
	}

	async forward(x) {
		this.x = x;

		const out = FloatArray(x.length);
		for (let i = 0; i < x.length; i++) {
			out[i] = this.f(x[i]);
		}
		return out;
	}

	async backward(grad) {
		const out = FloatArray(grad.length);
		for (let i = 0; i < grad.length; i++) {
			out[i] = this.fPrime(this.x[i]) * grad[i];
		}
		return out;
	}
}

class ReLU extends Activation {
	constructor() {
		super(
			x => x > 0 ? x : 0, 
			x => x > 0 ? 1 : 0
		);
	}
}

class Sigmoid extends Activation {
	constructor() {
		const sigmoid = x => 1 / (1 + Math.exp(-x));

		super(
			sigmoid, 
			x => {
				const s = sigmoid(x);
				return s * (1 - s);
			}
		);
	}
}

class Linear {
	constructor(inputLength, outputLength) {
		this.inputLength = inputLength;
		this.outputLength = outputLength;

		this.weights = createParams(outputLength * inputLength);
		this.biases = createParams(outputLength);
	}

	async forward(x) {
		this.x = x;

		const numSamples = x.length / this.inputLength;
		const out = FloatArray(numSamples * this.outputLength);

		for (let n = 0; n < numSamples; n++) {
			for (let o = 0; o < this.outputLength; o++) {
				const ni = n * this.outputLength + o;
				out[ni] = this.biases[o];
				for (let i = 0; i < this.inputLength; i++) {
					out[ni] += this.weights[o * this.inputLength + i] * x[n * this.inputLength + i];
				}
			}
		}

		return out;
	}

	async backward(grad) {
		let weightGrad = execute('linearWeightGrad', [this.inputLength, this.outputLength, this.x, grad]);
		let biasGrad = execute('sum', [grad, this.outputLength]);
		let inputGrad = execute('linearInputGrad', [this.inputLength, this.outputLength, this.weights, grad]);

		weightGrad = await weightGrad;
		biasGrad = await biasGrad;
		inputGrad = await inputGrad;

		updateParams(this.weights, weightGrad);
		updateParams(this.biases, biasGrad);

		return inputGrad;
	}
}

class MaxPool {
	constructor(inputSize, kernelSize) {
		this.inputSize = inputSize;
		this.inputLength = inputSize * inputSize;
		
		this.kernelSize = kernelSize;
		this.kernelLength = kernelSize * kernelSize;

		this.outputSize = Math.floor(inputSize / 2);
		this.outputLength = this.outputSize * this.outputSize;
	}

	async forward(x) {
		const numSamples = x.length / this.inputLength;

		const out = FloatArray(numSamples * this.outputLength);
		this.maxIndex = new Uint32Array(out.length);

		for (let n = 0; n < numSamples; n++) {
			for (let oy = 0; oy < this.outputSize; oy++) {
				for (let ox = 0; ox < this.outputSize; ox++) {
					const ni = n * this.outputLength + (oy * this.outputSize + ox);

					let max = -Infinity;
					let maxIndex = -1;

					for (let ky = 0; ky < this.kernelSize; ky++) {
						for (let kx = 0; kx < this.kernelSize; kx++) {
							const inputX = ox * this.kernelSize + kx;
							const inputY = oy * this.kernelSize + ky;

							const xi = n * this.inputLength + inputY * this.inputSize + inputX;
							const v = x[xi];
							if (v > max) {
								max = v;
								maxIndex = ky * this.kernelSize + kx;
							}
						}
					}

					out[ni] = max;
					this.maxIndex[ni] = maxIndex;
				}
			}
		}

		return out;
	}

	async backward(grad) {
		const numSamples = grad.length / this.outputLength;

		const inputGrad = FloatArray(numSamples * this.inputLength);

		for (let n = 0; n < numSamples; n++) {
			for (let oy = 0; oy < this.outputSize; oy++) {
				for (let ox = 0; ox < this.outputSize; ox++) {
					const gi = n * this.outputLength + (oy * this.outputSize + ox);

					const maxIndex = this.maxIndex[gi];
					const kx = maxIndex % this.kernelSize;
					const ky = (maxIndex - kx) / this.kernelSize;
					const ni = n * this.inputLength + (oy * this.kernelSize + ky) * this.inputSize + (ox * this.kernelSize + kx);

					inputGrad[ni] = grad[gi];
				}
			}
		}

		return inputGrad;
	}
}

function createParams(n, f = 1) {
	const out = FloatArray(n);
	for (let i = 0; i < out.length; i++) {
		out[i] = (Math.random() - 0.5) * f;
	}
	return out;
}

function FloatArray(n) {
	return new Float32Array(new SharedArrayBuffer(4 * n));
}

function softmax(x, outputLength) {
	const numSamples = x.length / outputLength;

	for (let i = 0; i < numSamples; i++) {
		let max = -Infinity;

		for (let j = 0; j < outputLength; j++) {
			const z = x[i * outputLength + j];
			z > max && (max = z);
		}

		let sum = 0;
		for (let j = 0; j < outputLength; j++) {
			const ni = i * outputLength + j;
			const e = Math.exp(x[ni] - max);
			x[ni] = e;
			sum += e;
		}

		sum = 1 / sum;
		for (let j = 0; j < outputLength; j++) {
			x[i * outputLength + j] *= sum;
		}
	}

	return x;
}

function crossEntropy(targets, predictions, outputLength) {
	const numSamples = targets.length / outputLength;
	
	let sum = 0;
	for (let i = 0; i < targets.length; i++) {
		const p = predictions[i];
		if (isFinite(p)) {
			sum += targets[i] * -Math.log(p + 1e-12);
		}
	}

	return sum / numSamples;
}

function softmaxCrossEntropyPrime(targets, predictions, outputLength) {
	const numSamples = targets.length / outputLength;

	const out = FloatArray(targets.length);			
	for (let i = 0; i < targets.length; i++) {
		out[i] = (predictions[i] - targets[i]) / numSamples;
	}
	
	return out;
}

function getAccuracy(targets, predictions, outputLength) {
	let correct = 0;

	for (let i = 0; i < targets.length; i += outputLength) {
		let max = -Infinity;
		let maxIndex = 0;
		for (let j = 0; j < outputLength; j++) {
			const prob = predictions[i + j];
			if (prob > max) {
				max = prob;
				maxIndex = j;
			}
		}

		if (targets[i + maxIndex] === 1) {
			correct++;
		}
	}

	const n = targets.length / outputLength;
	return correct / n;
}

// dataset

let data, datasets;

function parse(text) {
	data = [];

	const lines = text.split('\n');
	lines.shift();

	for (let line of lines) {
		line = line.trim();
		if (!line) continue;

		const items = line.split(',');
		const label = parseInt(items.shift());
		for (let i = 0; i < items.length; i++) {
			items[i] = parseInt(items[i]) / 255;
		}

		data.push({
			x: new Float32Array(items), 
			y: label
		});
	}

	data.sort(() => Math.random() - 0.5);

	console.log(`dataset loaded! (${data.length} samples)`);

	datasets = createDatasets(dataSplit, 0.8);
	train();
}

function createDatasets(dataSplit, trainSplit) {
	const partialData = data.slice(0, Math.floor(dataSplit * data.length));

	const n = Math.floor(trainSplit * partialData.length);
	const trainData = partialData.slice(0, n);
	const valData = partialData.slice(n);

	const train = prepareData(trainData);
	const val = prepareData(valData);

	return { train, val };
}

function prepareData(data, log = true) {
	const x = FloatArray(data.length * inputLength);
	const y = new Uint8Array(new SharedArrayBuffer(data.length * outputLength));

	const counter = {};

	for (let i = 0; i < data.length; i++) {
		const item = data[i];
		x.set(item.x, i * inputLength);
		y[i * outputLength + item.y] = 1;
	
		counter[item.y] = (counter[item.y] || 0) + 1;
	}

	if (log) {
		let text = `${data.length} total samples:\n`;

		for (const key in counter) {
			const n = counter[key];
			const percent = n / data.length * 100;
			text += `${key} / ${n} / ${percent.toFixed(2)}%\n`;
		}

		console.log(text);
	}

	return [x, y];
}

function inspect(layers) {
	let totalParams = 0;

	let text = `>>> ${layers.length} layers >>>\n`;

	for (let i = 0; i < layers.length; i++) {
		const layer = layers[i];
		text += `${layer.constructor.name}\n`;

		for (const key in layer) {
			const params = layer[key];
			if (ArrayBuffer.isView(params)) {
				totalParams += params.length;
				text += `  ${key}: ${params.length}\n`;
			}
		}
	}

	text += `total params: ${totalParams}\n`;

	console.log(text);
}

const dataSplit = 1;
const trainSplit = 0.8;
const epochs = 10;

const networks = {
	cnn: {
		batchSize: 1, 
		learningRate: 0.01, 
		layers: () => [
			new Conv(28, 1, 7, 16), 
			new ReLU(), 
			new MaxPool(22, 2), 
			new Linear(16 * 11 * 11, 10)
		]
	},
	nn: {
		batchSize: 16, 
		learningRate: 0.5, 
		layers: () => [
			new Linear(28 * 28, 4 * 4), 
			new ReLU(), 
			new Linear(4 * 4, 10)
		]
	} 
};

const network = networks.cnn;
const learningRate = network.learningRate;
const batchSize = network.batchSize;

const layers = network.layers();
inspect(layers);

const inputLength = layers[0].inputLength;
const outputLength = layers[layers.length - 1].outputLength;

async function train() {
	const [trainX, trainY] = datasets.train;
	const [valX, valY] = datasets.val;

	const trainCount = trainX.length / inputLength;

	for (let e = 0; e < epochs; e++) {
		const startTime = performance.now();

		for (let i = 0; i < trainCount; i += batchSize) {
			const batchStartTime = performance.now();

			const [batchX, batchY] = prepareData(data.slice(i, i + batchSize), false);

			const preds = await forward(batchX);
			await backward(batchY, preds);

			const f = Math.min(1, (i + batchSize) / trainCount);
			const accuracy = getAccuracy(batchY, preds, outputLength);
			console.log(`epoch ${e + 1}: ${(f * 100).toFixed(4)}%, acc: ${(accuracy * 100).toFixed(2)}%, time: ${((performance.now() - batchStartTime) / 1000).toFixed(3)}s`);

			await paramUpdate.promise;
		}

		const trainPreds = await forward(trainX);
		const trainLoss = crossEntropy(trainY, trainPreds, outputLength);
		const trainAccuracy = getAccuracy(trainY, trainPreds, outputLength);

		const valPreds = await forward(valX);
		const valLoss = crossEntropy(valY, valPreds, outputLength);
		const valAccuracy = getAccuracy(valY, valPreds, outputLength);

		const timeTaken = performance.now() - startTime;

		console.log(`epoch ${e + 1}, train loss: ${trainLoss.toFixed(3)}, train acc: ${(trainAccuracy * 100).toFixed(2)}%, val loss: ${valLoss.toFixed(3)}, val acc: ${(valAccuracy * 100).toFixed(2)}%, time taken: ${(timeTaken / 1000).toFixed(2)}s`);
	}
}

async function forward(x) {
	let y = x;
	for (let i = 0; i < layers.length; i++) {
		y = await layers[i].forward(y);
	}
	y = softmax(y, layers[layers.length - 1].outputLength);
	return y;
}

async function backward(targets, predictions) {
	let grad = softmaxCrossEntropyPrime(targets, predictions, outputLength);
	for (let i = layers.length - 1; i >= 0; i--) {
		grad = await layers[i].backward(grad);
	}
}