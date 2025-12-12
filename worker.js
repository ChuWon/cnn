if (isRunningInWorker()) {
	onmessage = event => onMessage(event.data);
} else {
	const { parentPort } = require('worker_threads');
	parentPort.on('message', onMessage);
	globalThis.postMessage = msg => parentPort.postMessage(msg);
}

function isRunningInWorker() {
	return typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope;
}

const handlers = {
	sum, 
	correlate, 
	convolve, 
	linearWeightGrad, 
	linearInputGrad
};

function onMessage(msg) {
	switch (msg.id) {
		case 'updateParams':
			updateParams(msg.params, msg.grad, msg.learningRate);
			postMessage({
				id: 'paramsUpdated'
			});
			break;

		case 'execute':
			const result = handlers[msg.name].apply(null, msg.args);
			postMessage({
				id: 'executed', 
				taskId: msg.taskId, 
				result
			});
			break;
	}
}

function updateParams(params, grad, learningRate) {
	for (let i = 0; i < params.length; i++) {
		params[i] -= grad[i] * learningRate;
	}
}

function sum(x, outputLength) {
	const numSamples = x.length / outputLength;

	const out = FloatArray(outputLength);
	for (let i = 0; i < outputLength; i++) {
		for (let n = 0; n < numSamples; n++) {
			out[i] += x[n * outputLength + i];
		}
	}

	return out;
}

function linearWeightGrad(inputLength, outputLength, x, grad) {
	const numSamples = grad.length / outputLength;
	const weightGrad = FloatArray(inputLength * outputLength);

	for (let i = 0; i < outputLength; i++) {
		for (let j = 0; j < inputLength; j++) {
			const ni = i * inputLength + j;
			for (let k = 0; k < numSamples; k++) {
				weightGrad[ni] += grad[i + k * outputLength] * x[j + k * inputLength];
			}
		}
	}

	return weightGrad;
}

function linearInputGrad(inputLength, outputLength, weights, grad) {
	const numSamples = grad.length / outputLength;
	const inputGrad = FloatArray(numSamples * inputLength);

	for (let i = 0; i < inputLength; i++) {
		for (let j = 0; j < numSamples; j++) {
			const ni = j * inputLength + i;
			for (let k = 0; k < outputLength; k++) {
				inputGrad[ni] += weights[k * inputLength + i] * grad[j * outputLength + k];
			}
		}
	}

	return inputGrad;
}

function correlate(inputSize, inputDepth, kernelSize, depth, x, kernels, biases) {
	const outputSize = inputSize - kernelSize + 1;
	const inputLength = inputDepth * inputSize * inputSize;
	const outputLength = depth * inputDepth * outputSize * outputSize;

	const numSamples = x.length / inputLength;
	const out = FloatArray(biases ? numSamples * outputLength : outputLength);

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
	const out = FloatArray(biases ? numSamples * outputLength : outputLength);

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

function FloatArray(n) {
	return new Float32Array(new SharedArrayBuffer(4 * n));
}
