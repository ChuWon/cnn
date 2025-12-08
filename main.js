/*

O = I - K + 1
grad = OxO
bias_grad = OxO
kernel_grad = KxK
input_grad = IxI

bias_grad = grad
kernel_grad = correlate(input, grad)
I - O + 1
I - (I - K + 1) + 1
I - I + K - 1 + 1 = K

input_grad = convolve(grad, kernel) = correlate(grad, rot180(kernel))
I - K + 1 + (K - 1) = I

convolve:
expand input by border of zeroes with thickness (kernelSize - 1) & compute correlation in reverse.
E = I + (K - 1) * 2
Y = E - K + 1 = I + 2*K - 2 - K + 1 = I + K - 1

*/

class Conv {
	constructor(inputSize, inputDepth, kernelSize, depth) {
		this.inputSize = inputSize;
		this.inputDepth = inputDepth;
		this.kernelSize = kernelSize;
		this.depth = depth;

		this.outputSize = inputSize - kernelSize + 1;
		this.inputLength = inputDepth * inputSize * inputSize;
		this.outputLength = this.depth * this.inputDepth * this.outputSize * this.outputSize;

		this.kernels = createParams(this.depth * this.inputDepth * this.kernelSize * this.kernelSize);
		this.biases = createParams(this.outputLength);
	}

	forward(x) {
		this.x = x;
		return correlate(this.inputSize, this.inputDepth, this.kernelSize, this.depth, x, this.kernels, this.biases);
	}

	backward(grad) {
		const numSamples = grad.length / this.outputLength;

		const kernelGrad = correlate(this.inputSize, this.inputDepth, this.outputSize, this.depth, this.x, grad);
		const inputGrad = convolve(this.outputSize, this.inputDepth, this.kernelSize, this.depth, grad, this.kernels);
		const biasGrad = new Float32Array(this.biases.length);

		for (let i = 0; i < this.outputLength; i++) {
			for (let n = 0; n < numSamples; n++) {
				biasGrad[i] += grad[n * this.outputLength + i];
			}
		}

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

	forward(x) {
		this.x = x;

		const out = new Float32Array(x.length);
		for (let i = 0; i < x.length; i++) {
			out[i] = this.f(x[i]);
		}
		return out;
	}

	backward(grad) {
		const out = new Float32Array(grad.length);
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

	forward(x) {
		this.x = x;

		const numSamples = x.length / this.inputLength;
		const out = new Float32Array(numSamples * this.outputLength);

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

	backward(grad) {
		const numSamples = grad.length / this.outputLength;

		const weightGrad = new Float32Array(this.weights.length);

		for (let i = 0; i < this.outputLength; i++) {
			for (let j = 0; j < this.inputLength; j++) {
				const ni = i * this.inputLength + j;
				for (let k = 0; k < numSamples; k++) {
					weightGrad[ni] += grad[i + k * this.outputLength] * this.x[j + k * this.inputLength];
				}
			}
		}

		const biasGrad = new Float32Array(this.biases.length);

		for (let i = 0; i < this.outputLength; i++) {
			for (let j = 0; j < numSamples; j++) {
				biasGrad[i] += grad[j * this.outputLength + i];
			}
		}

		const inputGrad = new Float32Array(numSamples * this.inputLength);

		for (let i = 0; i < this.inputLength; i++) {
			for (let j = 0; j < numSamples; j++) {
				const ni = j * this.inputLength + i;
				for (let k = 0; k < this.outputLength; k++) {
					inputGrad[ni] += this.weights[k * this.inputLength + i] * grad[j * this.outputLength + k];
				}
			}
		}

		updateParams(this.weights, weightGrad);
		updateParams(this.biases, biasGrad);

		return inputGrad;
	}
}

function createParams(n) {
	return Float32Array.from({ length: n }, () => Math.random() - 0.5);
}

function updateParams(params, grad) {
	for (let i = 0; i < params.length; i++) {
		params[i] -= grad[i] * learningRate;
	}
}

function correlate(inputSize, inputDepth, kernelSize, depth, x, kernels, biases) {
	const outputSize = inputSize - kernelSize + 1;
	const inputLength = inputDepth * inputSize * inputSize;
	const outputLength = depth * inputDepth * outputSize * outputSize;

	const numSamples = x.length / inputLength;
	const out = new Float32Array(biases ? numSamples * outputLength : outputLength);

	for (let n = 0; n < numSamples; n++) {
		for (let d = 0; d < depth; d++) {
			for (let i = 0; i < inputDepth; i++) {
				for (let oy = 0; oy < outputSize; oy++) {
					for (let ox = 0; ox < outputSize; ox++) {
						const bi = (d * inputDepth * outputSize * outputSize) + 
							(i * outputSize * outputSize) + 
							(oy * outputSize + ox);
						
						let ni = bi;
						if (biases) {
							ni += n * outputLength;
							out[ni] = biases?.[bi] || 0;
						}

						for (let ky = 0; ky < kernelSize; ky++) {
							for (let kx = 0; kx < kernelSize; kx++) {
								const xi = n * inputLength + 
									(i * inputSize * inputSize) + 
									(oy + ky) * inputSize + (ox + kx);
								const ki = (d * inputDepth * kernelSize * kernelSize) + 
									(i * kernelSize * kernelSize) + 
									(ky * kernelSize + kx);
								out[ni] += x[xi] * kernels[ki];
							}
						}
					}
				}
			}
		}
	}

	return out;
}

function convolve(inputSize, inputDepth, kernelSize, depth, x, kernels, biases) {
	const outputSize = inputSize + kernelSize - 1;
	const inputLength = inputDepth * inputSize * inputSize;
	const outputLength = depth * inputDepth * outputSize * outputSize;
	const borderSize = kernelSize - 1;

	const numSamples = x.length / inputLength;
	const out = new Float32Array(biases ? numSamples * outputLength : outputLength);

	for (let n = 0; n < numSamples; n++) {
		for (let d = 0; d < depth; d++) {
			for (let i = 0; i < inputDepth; i++) {
				for (let oy = 0; oy < outputSize; oy++) {
					for (let ox = 0; ox < outputSize; ox++) {
						const bi = (d * inputDepth * outputSize * outputSize) + 
							(i * outputSize * outputSize) + 
							(oy * outputSize + ox);
						
						let ni = bi;
						if (biases) {
							ni += n * outputLength;
							out[ni] = biases?.[bi] || 0;
						}

						for (let ky = 0; ky < kernelSize; ky++) {
							for (let kx = 0; kx < kernelSize; kx++) {
								const inputX = ox + kx - borderSize;
								const inputY = oy + ky - borderSize;

								if (inputX >= 0 && inputX < inputSize && inputY >= 0 && inputY < inputSize) {
									const xi = n * inputLength + 
										(i * inputSize * inputSize) + 
										inputY * inputSize + inputX;
									const ki = (d * inputDepth * kernelSize * kernelSize) + 
										(i * kernelSize * kernelSize) + 
										(kernelSize - 1 - ky) * kernelSize + (kernelSize - 1 - kx);
									out[ni] +=  x[xi] * kernels[ki];
								}
							}
						}
					}
				}
			}
		}
	}

	return out;
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

	const out = new Float32Array(targets.length);			
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

function convolveXD() {
	console.log('>>> TESTING CONVOLVE >>>');

	const x = [
		1, 2, 3, 4, 
		5, 6, 7, 8, 
		9, 10, 11, 12, 
		13, 14, 15, 16
	];

	const kernel = [
		1, -4, 2,  
		3, 5, 0, 
		3, 2, 2
	];

	const target = [
		1, -2, -3, -4, -10, 8, 
		8, -3, 12, 19, 2, 16, 
		27, 25, 55, 69, 28, 32, 
		55, 65, 111, 125, 56, 48, 
		66, 155, 186, 201, 126, 24, 
		39, 68, 99, 106, 62, 32
	];

	console.log(`INPUT: [${x}]`);
	console.log(`KERNEL: [${kernel}]`);
	console.log(`TARGET: [${target}]`);

	const inputSize = 4;
	const kernelSize = 3;
	const outputSize = inputSize + kernelSize - 1;
	const borderSize = kernelSize - 1;

	const out = new Float32Array(outputSize * outputSize);

	for (let oy = 0; oy < outputSize; oy++) {
		for (let ox = 0; ox < outputSize; ox++) {
			const ni = oy * outputSize + ox;

			for (let ky = 0; ky < kernelSize; ky++) {
				for (let kx = 0; kx < kernelSize; kx++) {
					const inputX = ox + kx - borderSize;
					const inputY = oy + ky - borderSize;

					if (inputX >= 0 && inputX < inputSize && inputY >= 0 && inputY < inputSize) {
						const xi = inputY * inputSize + inputX;
						const ki = (kernelSize - 1 - ky) * kernelSize + (kernelSize - 1 - kx);
						out[ni] +=  x[xi] * kernel[ki];
					}
				}
			}
		}
	}

	console.log(`OUTPUT: [${out}]`);

	const out2 = convolve(inputSize, 1, kernelSize, 1, x, kernel);
	console.log(`OUTPUT2: [${out2}]`);

	for (let i = 0; i < target.length; i++) {
		if (target[i] !== out[i] || target[i] !== out2[i]) {
			console.log(`TEST FAILED OwO!!!\nINDEX: ${i}\nTARGET: ${target[i]}\nVALUE: ${out[i]}\nVALUE2: ${out2[i]}`);
			return false;
		}
	}

	console.log(`TEST PASSED XD!`);
	return true;
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

	datasets = createDatasets(1, 0.8);
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

function prepareData(data) {
	const x = new Float32Array(data.length * inputLength);
	const y = new Uint8Array(data.length * outputLength);

	const counter = {};

	for (let i = 0; i < data.length; i++) {
		const item = data[i];
		x.set(item.x, i * inputLength);
		y[i * outputLength + item.y] = 1;
	
		counter[item.y] = (counter[item.y] || 0) + 1;
	}

	let text = `${data.length} total samples:\n`;

	for (const key in counter) {
		const n = counter[key];
		const percent = n / data.length * 100;
		text += `${key} / ${n} / ${percent.toFixed(2)}%\n`;
	}

	console.log(text);

	return [x, y];
}

const epochs = 15;
const batchSize = 16;

const dataSplit = 0.12;
const trainSplit = 0.8;
const learningRate = 0.5;

const networks = {
	cnn: () => [
		new Conv(28, 1, 3, 3), 
		new Sigmoid(), 
		new Linear(3 * 26 * 26, 10)
	], 
	nn: () => [
		new Linear(28 * 28, 4 * 4), 
		new ReLU(), 
		new Linear(4 * 4, 10)
	]
};

const layers = networks.cnn();

const inputLength = layers[0].inputLength;
const outputLength = layers[layers.length - 1].outputLength;

function train() {
	const partialData = data.slice(0, Math.floor(dataSplit * data.length));

	const n = Math.floor(trainSplit * partialData.length);
	const trainData = partialData.slice(0, n);
	const valData = partialData.slice(n);

	const [trainX, trainY] = datasets.train;
	const [valX, valY] = datasets.val;

	for (let e = 0; e < epochs; e++) {
		const startTime = performance.now();

		for (let i = 0; i < trainData.length; i += batchSize) {
			const batchX = trainX.slice(i * inputLength, (i + batchSize) * inputLength);
			const batchY = trainY.slice(i * outputLength, (i + batchSize) * outputLength);

			const preds = forward(batchX);
			backward(batchY, preds);

			/*const f = Math.min(1, (i + batchSize) / trainData.length);
			console.log(`epoch ${e + 1}: ${(f * 100).toFixed(2)}%`);*/
		}

		const trainPreds = forward(trainX);
		const trainLoss = crossEntropy(trainY, trainPreds, outputLength);
		const trainAccuracy = getAccuracy(trainY, trainPreds, outputLength);

		const valPreds = forward(valX);
		const valLoss = crossEntropy(valY, valPreds, outputLength);
		const valAccuracy = getAccuracy(valY, valPreds, outputLength);

		const timeTaken = performance.now() - startTime;

		console.log(`epoch ${e + 1}, train loss: ${trainLoss.toFixed(3)}, train acc: ${(trainAccuracy * 100).toFixed(2)}%, val loss: ${valLoss.toFixed(3)}, val acc: ${(valAccuracy * 100).toFixed(2)}%, time taken: ${(timeTaken / 1000).toFixed(2)}s`);
	}
}

function forward(x) {
	let y = x;
	for (let i = 0; i < layers.length; i++) {
		y = layers[i].forward(y);
	}
	y = softmax(y, layers[layers.length - 1].outputLength);
	return y;
}

function backward(targets, predictions) {
	let grad = softmaxCrossEntropyPrime(targets, predictions, outputLength);
	for (let i = layers.length - 1; i >= 0; i--) {
		grad = layers[i].backward(grad);
	}
}

convolveXD();

const file = 'mnist_train.csv';

if (typeof window === 'undefined') {
	const fs = require('fs');
	const text = fs.readFileSync(file, { encoding: 'utf8' });
	parse(text);
} else {
	fetch(file).then(res => res.text()).then(parse);
}