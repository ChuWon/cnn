const DEV = window.location.hostname === 'localhost';

const worker = new Worker('./webWorker.js');

let progress = 0;
let progressText = 'idle';

let loaded = false;

let modelId;
function createModel() {
	reset();
	worker.postMessage({
		id: 'createModel', 
		modelId
	});

	for (const key in defaultSettings) {
		setSetting(key, defaultSettings[key]);
	}
}

modelId = '69969';
worker.postMessage({
	id: 'createModel', 
	modelId
});

let predictT = 0;

function reset(id) {
	modelId = id || Math.random().toString(32).slice(2);
	epoch = 0;
	epochPercent = 0;
	lastCheckpointT = 0;
	setEpochProgress();

	epoching = false;
	lossCurving = false;
	lossLandscaping = false;
	predicting = false;
	predictT = 1;
	prediction = null;

	lossLandscapePoints.length = 0;
	disposeLossLandscape();

	resetGraphs();
}

const lossCurveLength = 69;
const lossCurveRange = [-8.555, 4.666];

const lossLandscapeLength = 36;
const lossLandscapeSize = lossLandscapeLength * lossLandscapeLength;
const lossLandscapeRange = [-6.9, 6.9];

let epoch = 0;
let epochPercent = 0;
let lastCheckpointT = 0;

let epoching = false;
let lossCurving = false;
let lossLandscaping = false;
let predicting = false;

function setEpochProgress() {
	progress = Math.min(1, (epoch + epochPercent) / epochs);
	progressText = `training epoch ${epoch}/${epochs}${epochPercent ? ` (${(epochPercent * 100).toFixed(2)}%)` : ''}`;
}

function setEpoch(msg) {
	epoch = msg.epoch;
	epochPercent = msg.epochPercent;
	setEpochProgress();
}

let prediction;

worker.onmessage = function (event) {
	const msg = event.data;

	switch (msg.id) {
		case 'progress':
			progress = msg.percent;
			progressText = `loading dataset ${(msg.percent * 100).toFixed(2)}%`;
			break;

		case 'loaded':
			loaded = true;
			createModel();
			createDatasets();
			setLearningRate();
			setBatchSize();

			browseUI.reqContent();

			fetch('cnn-e10-11.19.666')
				.then(res => res.text())
				.then(importCheckpoint)
				.catch(error => console.log(`checkpoint load error: ${error.message}`))
			break;

		case 'failed':
			alert(`Failed to load dataset, can't train model. Reload the page to retry.`);
			break;

		case 'dataset':
			browseUI.setContent(msg);
			break;

		case 'epochBatch':
			epoching = false;
			if (msg.modelId !== modelId) return;
			setEpoch(msg);
			addGraph('batchTime', msg.batchTimeTaken);
			setGraph('epochTime', msg.epochTimeTaken);

			if (settings.autoSaveCheckpoint) {
				const t = msg.epoch + msg.epochPercent;
				if (t - lastCheckpointT > settings.checkpointSaveInterval) {
					lastCheckpointT = t;
					saveCheckpoint();
				}
			}
			break;

		case 'epoch':
			epoching = false;
			if (msg.modelId !== modelId) return;
			setEpoch(msg);
			addGraph('epochTime', msg.epochTimeTaken);
			addGraph('trainLoss', msg.trainLoss);
			addGraph('trainAccuracy', msg.trainAccuracy);
			addGraph('valLoss', msg.valLoss);
			addGraph('valAccuracy', msg.valAccuracy);
			break;

		case 'prediction': {
			predicting = false;

			updateStartActivation();

			for (let i = msg.layerOutputs.length - 1; i >= 0; i--) {
				const outputs = msg.layerOutputs[i];
				const layer = layers[i];
				if (layer.type === 'Activation') {
					i--;
				}

				let min = Infinity;
				let max = -Infinity;
				for (let j = 0; j < outputs.length; j++) {
					const v = outputs[j];
					min = Math.min(min, v);
					max = Math.max(max, v);
				}

				const f = max !== min ? 1 / (max - min) : 1;
				for (let j = 0; j < outputs.length; j++) {
					activationData[layer.offset + j] = (outputs[j] - min) * f;
				}
			}

			gl.bindBuffer(gl.ARRAY_BUFFER, activationBuffer);
			gl.bufferSubData(gl.ARRAY_BUFFER, 0, activationData);

			const probs = msg.layerOutputs[msg.layerOutputs.length - 1];

			let maxProb = -Infinity;
			let result = -1;
			for (let i = 0; i < probs.length; i++) {
				const p = probs[i];
				if (p > maxProb) {
					result = i;
					maxProb = p;
				}
			}

			prediction = {
				probs,
				maxProb,  
				result
			};
		}	break;

		case 'elPrediction':
			const el = document.getElementById(msg.el);
			el.innerText = `pred: ${msg.y}`;
			break;

		case 'lossCurve':
			lossCurving = false;
			if (msg.modelId !== modelId) break;
			addGraph('lossCurve', msg.value);
			break;

		case 'lossLandscape': 
			lossLandscaping = false;
			if (msg.modelId !== modelId) break;
			lossLandscapePoints.push(msg.value);
			updateLossLandscape();
			break;

		case 'checkpoint':
			const a = document.createElement('a');
			a.href = URL.createObjectURL(new Blob([msg.json], { type: 'text/plain' }));
			a.download = `cnn-e${epoch}-${(epochPercent * 100).toFixed(2)}.666`;
			a.click();
			break;

		case 'checkpointData':
			reset(msg.modelId);
			setEpoch(msg);
			lastCheckpointT = epoch + epochPercent;

			const data = msg.data;

			for (const key in data.settings) {
				const v = data.settings[key];
				if (typeof settings[key] === typeof v) {
					setSetting(key, v);
				}
			}

			setSetting('trainingEnabled', false);
			setSetting('lossLandscape', false);

			for (const key in data.graphs) {
				if (key in graphs) {
					const list = data.graphs[key];
					for (let i = 0; i < list.length; i++) {
						addGraph(key, list[i]);
					}
				}
			}

			if ((data.lossLandscape?.length || 0) > 0) {
				for (let i = 0; i < data.lossLandscape.length; i++) {
					lossLandscapePoints.push(data.lossLandscape[i]);
				}
				updateLossLandscape();
			}
			break;

		case 'checkpointError':
			alert(`Failed to load checkpoint!\nError: ${msg.error}`);
			break;

		case 'layers':
			layers = msg.layers;
			layers.unshift({
				type: 'Input', 
				size: inputSize, 
				depth: 1
			});
			initObjects();
			break;

		case 'params':
			for (const i in msg.layers) {
				Object.assign(layers[i], msg.layers[i]);
			}
			break;

		default:
			console.log(`Unknown msg id from worker: ${msg.id}`);
	}
}

function updateStartActivation() {
	for (let i = 0; i < startActivationData.length; i++) {
		startActivationData[i] = startActivationData[i] + (activationData[i] - startActivationData[i]) * predictT;
	}
	predictT = 0;

	gl.bindBuffer(gl.ARRAY_BUFFER, startActivationBuffer);
	gl.bufferSubData(gl.ARRAY_BUFFER, 0, startActivationData);
}

function setLearningRate() {
	worker.postMessage({
		id: 'setLearningRate', 
		value: settings.learningRate
	});
}

function setBatchSize() {
	worker.postMessage({
		id: 'setBatchSize', 
		value: settings.batchSize
	});
}

let datasetNeedsUpdate = false;
function createDatasets() {
	datasetNeedsUpdate = true;
}

function saveCheckpoint() {
	const data = {
		settings, 
		lossLandscape: new Float32Array(lossLandscapePoints), 
		graphs: {}
	}

	for (const key in graphs) {
		data.graphs[key] = new Float32Array(graphs[key].points);
	}

	worker.postMessage({
		id: 'checkpointData', 
		data
	});
}

function importCheckpoint(json) {
	worker.postMessage({
		id: 'checkpoint', 
		json
	});
}

let userInput;

function predict(image) {
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = inputSize;
	const ctx = canvas.getContext('2d');

	ctx.drawImage(image, 0, 0, inputSize, inputSize);

	const imageData = ctx.getImageData(0, 0, inputSize, inputSize);

	userInput = new Float32Array(inputLength);
	for (let i = 0; i < inputLength; i++) {
		userInput[i] = imageData.data[i * 4 + 3] / 255;
	}
}

function isTraining() {
	return settings.trainingEnabled && (epoch < epochs || settings.endlessTraining);
}

let lossLandscape;
const lossLandscapePoints = [];

function updateLossLandscape() {
	disposeLossLandscape();

	const n = lossLandscapePoints.length;
	if (n <= 0) return;

	let min = Infinity;
	let max = -Infinity;
	for (const loss of lossLandscapePoints) {
		min = Math.min(loss, min);
		max = Math.max(loss, max);
	}

	const points = [];
	const pointData = new Float32Array(n * 3);

	for (let i = 0; i < lossLandscapePoints.length; i++) {
		const f = (lossLandscapePoints[i] - min) / (max - min);
		const p = [
			((i % lossLandscapeLength) / (lossLandscapeLength - 1) * 2 - 1) * 75, 
			f * 100 - 80, 
			(Math.floor(i / lossLandscapeLength) / (lossLandscapeLength - 2) * 2 - 1) * 75
		];
		p.f = f;
		pointData.set(p, i * 3);
		points.push(p);
	}

	const posData = [];
	const intensityData = [];
	const lineData = [];
	const normalData = [];

	const vertexNormals = {};

	const h = Math.floor(n / lossLandscapeLength) - 1;
	
	for (let i = 0, l = Math.min(points.length, lossLandscapeLength) - 1; i < l; i++) {
		lineData.push(...points[i], ...points[i + 1]);
	}

	for (let y = 0; y <= h; y++) {
		const w = (y < h ? lossLandscapeLength : n % lossLandscapeLength) - 2;
		for (let x = 0; x <= w; x++) {
			const a = points[y * lossLandscapeLength + x];
			const b = points[y * lossLandscapeLength + x + 1];
			const c = points[(y + 1) * lossLandscapeLength + x];
			const d = points[(y + 1) * lossLandscapeLength + x + 1];

			y > 0 && a && b && lineData.push(...a, ...b);
			a && c && lineData.push(...a, ...c);
			x === w && b && d && lineData.push(...b, ...d);
			y >= h - 1 && c && d && lineData.push(...c, ...d);

			if (a && b && c && d) {
				posData.push(
					...b, ...a, ...c,
					...b, ...c, ...d
				);
				intensityData.push(
					b.f, a.f, c.f, 
					b.f, c.f, d.f
				);

				const n1 = cross(dir(c, a), dir(b, a));
				const n2 = cross(dir(d, c), dir(b, c));
				normalData.push(
					...n1, ...n1, ...n1, 
					...n2, ...n2, ...n2
				);

				addVertexNormal(b, n1);
				addVertexNormal(a, n1);
				addVertexNormal(c, n1);

				addVertexNormal(b, n2);
				addVertexNormal(c, n2);
				addVertexNormal(d, n2);
			}
		}
	}

	const vertexNormalData = new Float32Array(posData.length);

	for (let i = 0; i < posData.length; i += 3) {
		const key = vertexToKey(posData[i], posData[i + 1], posData[i + 2]);
		const vn = vertexNormals[key];
		const f = 1 / vn.count;
		vertexNormalData[i] = vn.x * f;
		vertexNormalData[i + 1] = vn.y * f;
		vertexNormalData[i + 2] = vn.z * f;
	}

	function addVertexNormal(v, normal) {
		const key = vertexToKey(v[0], v[1], v[2]);
		let vn = vertexNormals[key];
		if (!vn) {
			vertexNormals[key] = vn = {
				x: 0, 
				y: 0, 
				z: 0, 
				count: 0
			};
		}

		vn.x += normal[0];
		vn.y += normal[1];
		vn.z += normal[2];
		vn.count++;
	}

	function vertexToKey(x, y, z) {
		return `${toKey(x)},${toKey(y)},${toKey(z)}`;
	}

	function toKey(x) {
		return Math.floor(x * 1e5);
	}

	lossLandscape = {
		points, 
		pointBuffer: createBuffer(pointData), 
		posBuffer: createBuffer(new Float32Array(posData)), 
		normalBuffer: createBuffer(new Float32Array(normalData)), 
		vertexNormalBuffer: createBuffer(vertexNormalData), 
		intensityBuffer: createBuffer(new Float32Array(intensityData)), 
		vertexCount: posData.length / 3, 
		lineBuffer: createBuffer(new Float32Array(lineData)), 
		lineCount: lineData.length / 3
	};
}

function dir(a, b) {
	let x = a[0] - b[0];
	let y = a[1] - b[1];
	let z = a[2] - b[2];

	let d = Math.hypot(x, y, z);
	if (d > 0) {
		d = 1 / d;
		x *= d;
		y *= d;
		z *= d;
	}

	return [x, y, z];
}

function cross(a, b) {
	return [
		a[1] * b[2] - a[2] * b[1], 
		a[2] * b[0] - a[0] * b[2], 
		a[0] * b[1] - a[1] * b[0]
	];
}

function disposeLossLandscape() {
	if (lossLandscape) {
		gl.deleteBuffer(lossLandscape.pointBuffer);
		gl.deleteBuffer(lossLandscape.posBuffer);
		gl.deleteBuffer(lossLandscape.intensityBuffer);
		gl.deleteBuffer(lossLandscape.lineBuffer);
		lossLandscape = null;
	}
}

const epochs = 5;

const inputSize = 28;
const inputLength = 28 * 28;

function Grid(size = 20) {
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext('2d');

	ctx.fillStyle = '#151515';
	ctx.fillRect(0, 0, size, size);

	ctx.beginPath();
	const s = size / 2;
	ctx.moveTo(s, 0);
	ctx.lineTo(s, size);
	ctx.moveTo(0, s);
	ctx.lineTo(size, s);
	ctx.lineWidth = size * 0.04;
	ctx.strokeStyle = 'hsla(0, 0%, 100%, 0.02)';
	ctx.stroke();

	return canvas;
}

let canAutoPredict = true;

function SketchUI() {
	const size = 150;
	const el = fromHtml(`<div style="
		position: absolute;
		right: 12px;
		bottom: 12px;
		display: flex;
		flex-direction: column;
		align-items: end;
		grid-gap: 5px;
		pointer-events: all;
	">
		<div class="row">
			<div class="btn browse-btn">browse</div>
			<div class="btn clear-btn">clear</div>
		</div>
		<canvas style="
			width: ${size}px;
			height: ${size}px;
			border-radius: 5px;
			background: hsla(0, 0%, 20%, 0.3);
			border: 2px solid hsla(0, 0%, 100%, 0.2);
			pointer-events: all;
		"></canvas>
		<div>draw a digit xp</div>
	</div>`);
	uiEl.appendChild(el);

	const canvas = el.querySelector('canvas');
	canvas.width = canvas.height = size * window.devicePixelRatio;
	const ctx = canvas.getContext('2d');

	el.querySelector('.browse-btn').onclick = function () {
		browseUI.setVisible(true);
	}

	el.querySelector('.clear-btn').onclick = function () {
		paths.length = 0;
		path = null;
		draw();

		canAutoPredict = true;
		predictT = 1;
		userInput = null;
	}

	const paths = [];

	let path;
	canvas.onmousedown = function (event) {
		if (event.button === 0 && !path) {
			onPointerStart(event);
		}
	}
	window.addEventListener('mousemove', onPointerMove);
	window.addEventListener('mouseup', event => {
		if (event.button === 0) {
			path = null;
		}
	});

	function onPointerStart(event) {
		path = [getPointer(event)];
		paths.push(path);
		draw();

		canAutoPredict = false;
		predict(canvas);
	}

	function onPointerMove(event) {
		if (path) {
			path.push(getPointer(event));
			draw();

			canAutoPredict = false;
			predict(canvas);
		}
	}

	canvas.onTouchStart = function (touch) {
		if (!path) {
			onPointerStart(touch);
			const id = touch.identifier;
			onTouchMove[id] = onPointerMove;
			onTouchEnd[id] = function () {
				path = null;
			}
		}
	}

	function draw() {
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		ctx.save();
		ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

		ctx.filter = 'blur(2px)';
		
		ctx.beginPath();
		for (const path of paths) {
			ctx.moveTo(...path[0]);
			ctx.lineTo(...path[0]);
			for (let i = 1; i < path.length; i++) {
				ctx.lineTo(...path[i])
			}
		}

		ctx.lineWidth = 15;
		ctx.strokeStyle = '#fff';
		ctx.lineCap = ctx.lineJoin = 'round';
		ctx.stroke();

		ctx.restore();
	}

	function getPointer(event) {
		const box = canvas.getBoundingClientRect();
		return [
			(event.clientX - box.x) / box.width * size, 
			(event.clientY - box.y) / box.height * size
		];
	}

	return el;
}

// ui

const PI2 = Math.PI * 2;

const canvas = document.getElementById('canvas');

const options = {
	antilias: true, 
	alpha: true
};
const gl = canvas.getContext('webgl', options) || canvas.getContext('experimental-webgl', options);

if (!gl) {
	alert(`webgl not supported. get better device LOL XD`);
	throw new Error('webgl not supported');
}

const ext = gl.getExtension('ANGLE_instanced_arrays');
if (!ext) {
	alert(`ext not supported RIP XD LOL no nn 4u XD`);
	throw new Error('ext not supported');
}

gl.enable(gl.DEPTH_TEST);
gl.enable(gl.CULL_FACE);
gl.cullFace(gl.BACK);

gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

const program = createProgram(`

precision highp float;

attribute vec3 position;
attribute vec3 normal;
attribute vec3 worldPos;
attribute float startActivation;
attribute float activation;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform vec3 lightPos;
uniform float t;

varying vec3 vColor;

void main() {
	vec3 p = position + worldPos;
	gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);

	float light = max(0.0, dot(normalize(lightPos - p), normal)) * 0.4 + 0.6;
	vColor = vec3(startActivation + (activation - startActivation) * t) * light;
}

`, `

precision highp float;

varying vec3 vColor;

void main() {
	gl_FragColor = vec4(vColor, 1.0);
}

`);

const lineProgram = createProgram(`

precision highp float;

attribute vec3 position;
attribute vec3 worldPos;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

void main() {
	vec3 p = position + worldPos;
	gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
	gl_Position.z -= 0.005;
}

`, `

precision highp float;

void main() {
	gl_FragColor = vec4(0.2);
}

`);

const bgProgram = createProgram(`

precision highp float;

attribute vec3 position;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

varying vec3 vPos;

void main() {
	gl_Position = projectionMatrix * viewMatrix * vec4(position * 5000.0, 1.0);
	gl_Position.z = 0.0;
	vPos = position;
}

`, `

precision highp float;

uniform sampler2D map;

varying vec3 vPos;

void main() {
	vec3 p = vPos * 2.0;
	vec2 uv = abs(p.z) > 0.999 ? p.xy : (abs(p.x) > 0.999 ? p.zy : p.xz);
	gl_FragColor = texture2D(map, uv * 20.0);
}

`);

const planeProgram = createProgram(`

precision highp float;

attribute vec3 position;
attribute float intensity;
attribute vec3 normal;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

varying vec3 vPos;
varying vec3 vViewPos;
varying vec3 vNormal;
varying float vIntensity;

void main() {
	vec4 p = viewMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * p;

	vPos = position;
	vViewPos = p.xyz;
	vNormal = normal;
	vIntensity = intensity;
}

`, `

precision highp float;

uniform vec3 lightPos;
uniform float t;
uniform vec3 center;

varying vec3 vPos;
varying vec3 vViewPos;
varying vec3 vNormal;
varying float vIntensity;

float PI2 = ${PI2.toString()}2666;

void main() {
	vec3 viewDir = -normalize(vViewPos);
	float spec = pow(max(dot(reflect(normalize(vViewPos - lightPos), vNormal), viewDir), 0.0), 0.7);
	float fresnel = pow(1.0 - abs(dot(vNormal, viewDir)), 7.0);
	float diffuse = max(0.0, dot(normalize(lightPos - vPos), vNormal));
	float light = 0.3 + diffuse * 0.7 + spec * 0.5 + fresnel * 0.22;
	vec3 color = mix(vec3(0.01, 0.66, 0.95), vec3(0.54, 0.81, 0.22), vIntensity);

	float r = t * 26.66;
	vec2 p = vPos.xz - center.xz;
	if (length(p) < r) {		
		float n = 10.0;
		float i = floor(mod((atan(p.y, p.x) + PI2), PI2) / PI2 * n);
		float step = PI2 / n;
		float a0 = i * step;
		float a1 = a0 + step;
	
		float d = 0.4 + t * 0.3;
		float m = mod(i, 2.0);
		vec2 p1 = vec2(cos(a0), sin(a0)) * (1.0 + d * (m - 1.0)) * r;
		vec2 p0 = vec2(cos(a1), sin(a1)) * (1.0 - d * m) * r;
		vec2 mid = (p0 + p1) * 0.5;
		vec2 nor = vec2(-(p1.y - p0.y), p1.x - p0.x);
		float proj = dot(nor, p - mid);
	
		if (proj < 0.0) {
			color = mix(color, vec3(1.0), t);
		}
	}

	color *= light;
	gl_FragColor = vec4(color, 1.0);
}

`);

const planeLineProgram = createProgram(`

precision highp float;

attribute vec3 position;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;

void main() {
	gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
	gl_Position.z -= 0.02;
}

`, `

precision highp float;

uniform vec4 color;
	
void main() {
	gl_FragColor = color;
}

`);

const lightPos = [-420, 666, 699.69];

const mesh = {
	indices: [0, 2, 1, 2, 3, 1, 4, 6, 5, 6, 7, 5, 8, 10, 9, 10, 11, 9, 12, 14, 13, 14, 15, 13, 16, 18, 17, 18, 19, 17, 20, 22, 21, 22, 23, 21],
	vertices: [0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
	normals: [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1], 
	lines: getBoxLines()
};

function getBoxLines() {
	const poly = [
		[-0.5, 0.5], 
		[0.5, 0.5], 
		[0.5, -0.5], 
		[-0.5, -0.5], 
	];

	const list = [];

	for (let i = 0; i < poly.length; i++) {
		const a = poly[i];
		const b = poly[(i + 1) % poly.length];
		list.push(...a, 0.5, ...b, 0.5);
		list.push(...a, -0.5, ...b, -0.5);
		list.push(...a, 0.5);
		list.push(...a, -0.5);
	}

	return list;
}

const map = gl.createTexture();
gl.activeTexture(gl.TEXTURE0);
gl.bindTexture(gl.TEXTURE_2D, map);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, Grid(32));
gl.generateMipmap(gl.TEXTURE_2D);

const posBuffer = createBuffer(new Float32Array(mesh.vertices));
const normalBuffer = createBuffer(new Float32Array(mesh.normals));
const lineBuffer = createBuffer(new Float32Array(mesh.lines));

const indexBuffer = gl.createBuffer();
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.indices), gl.STATIC_DRAW);

let worldPosData, 
	startActivationData, 
	activationData;

let worldPosBuffer, 
	startActivationBuffer, 
	activationBuffer;

let objectCount = 0;

const gap = 1.666;
const layerGap = 25;
const depthGap = 2.666;

let layers = [{
	type: 'Input', 
	size: 1, 
	depth: 1
}];

const texts = [];

function initObjects() {
	if (worldPosBuffer) {
		gl.deleteBuffer(worldPosBuffer);
		gl.deleteBuffer(startActivationBuffer);
		gl.deleteBuffer(activationBuffer);
	}

	boxes = [];
	texts.length = 0;

	let pz = 0;
	let lastDepth = 1;

	function addBoxes(size, depth) {
		lastDepth = depth;

		for (let z = 0; z < depth; z++) {
			for (let y = 0; y < size; y++) {
				for (let x = 0; x < size; x++) {
					boxes.push([
						getCoord(x, size), 
						-getCoord(y, size), 
						pz
					]);
				}
			}

			pz += depthGap;
		}

		pz += layerGap;
	}

	for (let i = 0; i < layers.length; i++) {
		const layer = layers[i];
		layer.z = pz;
		layer.offset = boxes.length;

		switch (layer.type) {
			case 'Input':
				addBoxes(layer.size, layer.depth);
				layer.outputSize = layer.size;
				break;

			case 'Conv':
				addBoxes(layer.outputSize, layer.depth);
				break;

			case 'MaxPool':
				addBoxes(layer.outputSize, lastDepth);
				break;

			case 'Linear':
				const size = Math.sqrt(layer.outputLength);
				if (Math.floor(size) !== size) {
					for (let y = 0; y < layer.outputLength; y++) {
						boxes.push([
							0, 
							-getCoord(y, layer.outputLength), 
							pz
						]);
					}
					
					pz += layerGap;
				} else {
					addBoxes(size, 1);
				}
				break;

			case 'Activation':
				layer.offset = layers[i - 1].offset;
				break;
		}

		if (layer.kernelSize) {
			texts.push({
				text: `${layer.type} ${layer.kernelSize}x${layer.kernelSize}${layer.depth ? `x${layer.depth}` : ''}`, 
				pos: [
					0, 
					(layer.outputSize + 1) * gap, 
					layer.z + (lastDepth - 1) * depthGap / 2
				]
			});
		}
	}

	if (boxes.length > 0) {
		const centerZ = (boxes[boxes.length - 1][2] + boxes[0][2]) / 2;
		for (const box of boxes) {
			box[2] -= centerZ;
		}

		for (const layer of layers) {
			layer.z -= centerZ;
		}

		for (const item of texts) {
			item.pos[2] -= centerZ;
		}
	}

	objectCount = boxes.length;
	worldPosData = new Float32Array(3 * objectCount);
	startActivationData = new Float32Array(objectCount);
	activationData = new Float32Array(objectCount);

	for (let i = 0; i < objectCount; i++) {
		worldPosData.set(boxes[i], i * 3);
	}

	worldPosBuffer = createBuffer(worldPosData);
	startActivationBuffer = createBuffer(startActivationData, true);
	activationBuffer = createBuffer(activationData, true);
}

initObjects();

function getCoord(x, n) {
	return n > 1 ? (x / (n - 1) * 2 - 1) * n * gap : 0;
}

const hudCanvas = document.getElementById('hudCanvas');
const hudCtx = hudCanvas.getContext('2d');

resizeCanvas();

const uiEl = document.querySelector('.ui');
const headerEl = document.querySelector('.header');
const sketchEl = SketchUI();

function resizeCanvas() {
	canvas.width = window.innerWidth * window.devicePixelRatio;
	canvas.height = window.innerHeight * window.devicePixelRatio;

	hudCanvas.width = canvas.width;
	hudCanvas.height = canvas.height;
}

function resize() {
	resizeCanvas();

	const scale = Math.max(window.innerWidth / 1366, window.innerHeight / 768);

	Object.assign(uiEl.style, {
		transform: `scale(${scale})`, 
		width: window.innerWidth / scale + 'px', 
		height: window.innerHeight / scale + 'px', 
	});
}

window.onresize = function () {
	resize();
	render();
}
resize();

const settingsEl = document.querySelector('.settings');

const settings = {
	trainingEnabled: false, 
	endlessTraining: false, 
	lossLandscape: false, 
	autoSaveCheckpoint: false, 
	learningRate: [0.01, 0.01, 1, 0.01], 
	checkpointSaveInterval: [0.1, 0.01, 1, 0.01], 
	batchSize: [1, 1, 666, 1], 
	trainSplit: [0.8, 0.01, 0.99, 0.01], 
	dataSplit: [1, 0.01, 1, 0.01], 
	orbitSpeed: [1, 0, 50, 1]
};

const settingOnChange = {
	learningRate: setLearningRate, 
	trainSplit: createDatasets, 
	dataSplit: createDatasets, 
	batchSize: setBatchSize
};

for (const key in settings) {
	const value = settings[key];

	if (Array.isArray(value)) {
		const [n, min, max, step] = value;
		settings[key] = n;

		const el = fromHtml(`<div class="row">
			<div>${fromCamel(key)}:</div>
			<input type="range" class="range" min="${min}" max="${max}" step="${step}" id="${key}">
			<div></div>
		</div>`);

		const rangeEl = el.querySelector('.range');
		rangeEl.value = n;
		rangeEl.nextElementSibling.innerText = n;
		rangeEl.onchange = function () {
			settings[key] = parseFloat(this.value);
			this.nextElementSibling.innerText = settings[key];
			settingOnChange[key] && settingOnChange[key](settings[key]);
		}
		rangeEl.oninput = function () {
			this.nextElementSibling.innerText = this.value;
		}

		settingsEl.appendChild(el);
	} else {
		const el = fromHtml(`<label class="row">
			<input type="checkbox" class="checkbox" id="${key}">
			<div>${fromCamel(key)}</div>
		</label>`);

		const checkboxEl = el.querySelector('.checkbox');
		checkboxEl.checked = value;

		checkboxEl.onchange = function () {
			settings[key] = this.checked;
			settingOnChange[key] && settingOnChange[key](settings[key]);
		}

		settingsEl.appendChild(el);
	}
}

const defaultSettings = Object.assign({}, settings);

function setSetting(key, value) {
	settings[key] = value;
	const el = document.getElementById(key);
	el[el.checked !== undefined ? 'checked' : 'value'] = value;
	el.onchange();
}

function fromCamel(text){
	text = text.replace(/([A-Z])/g,' $1');
	return text.charAt(0).toUpperCase() + text.slice(1);
}

function fromHtml(html) {
	const div = document.createElement('div');
	div.innerHTML = html;
	return div.children[0];
}

const btnsEl = fromHtml(`<div class="row" style="margin-top: 3px;">
	<div class="btn reset-btn">reset</div>
	<div class="btn export-btn">export</div>
	<div class="btn import-btn">import</div>
</div>`);
settingsEl.appendChild(btnsEl);

btnsEl.querySelector('.reset-btn').onclick = createModel;

btnsEl.querySelector('.export-btn').onclick = saveCheckpoint;

btnsEl.querySelector('.import-btn').onclick = function () {
	const el = document.createElement('input');
	el.type = 'file';
	el.accept = '.666';

	el.oninput = function (event) {
		const file = this.files[0];

		file && importFile(file);
	}

	el.click();
}

let dragT = 0;

document.ondragover = event => event.preventDefault();
document.ondrop = function (event) {
	event.preventDefault();
	const file = event.dataTransfer.files[0];
	file && importFile(file);
	dragging = false;
}

let dragging = false;
document.ondragenter = function () {
	dragging = true;
}
document.ondragleave = function () {
	dragging = false;
}

function importFile(file) {
	const reader = new FileReader();
	reader.onload = function () {
		importCheckpoint(this.result);
	}
	reader.readAsText(file);
}

// rendering

CanvasRenderingContext2D.prototype.scale2 = function (f) {
	this.scale(f, f);
}

HTMLElement.prototype.setClass = function (cls, v) {
	this.classList[v ? 'add' : 'remove'](cls);
}

const colors = {
	activation: '#ffeb3b', 
	label: '#fb382a'
};

let graphs;

function initGraphs() {
	graphs = {};

	const list = ['trainLoss', 'trainAccuracy', 'valLoss', 'valAccuracy', 'batchTime', 'epochTime', 'lossCurve'];

	for (let i = 0; i < list.length; i++) {
		const key = list[i];
		graphs[key] = {
			name: fromCamel(key), 
			points: [], 
			max: -Infinity,
			i: 1 + i, 
			visible: true
		};
	}

	graphs.epochTime.formatter = formatTime;
	graphs.batchTime.formatter = formatTime;
}

function formatTime(s) {
	const m = Math.floor(s / 60);
	s = s % 60;
	return m > 0 ? `${m}m ${Math.floor(s)}s` : `${s.toFixed(2)}s`;
}

function addGraph(name, y) {
	if (!isFinite(y)) y = 0;
	const graph = graphs[name];
	graph.points.push(y);
	graph.max = Math.max(graph.max, y);

	if (graph.points.length > 4666) {
		graph.points.shift();
		graph.max = Math.max.apply(Math, graph.points);
	}
}

function setGraph(name, y) {
	if (!isFinite(y)) y = 0;
	const graph = graphs[name];
	graph.points[Math.max(graph.points.length - 1, 0)] = y;
	graph.max = Math.max(graph.max, y);
}

function resetGraphs() {
	for (const key in graphs) {
		const graph = graphs[key];
		graph.points = [];
		graph.max = -Infinity;
	}
}

initGraphs();

const troll = new Image();
troll.src = 'troll.png';
troll.onload = () => console.log('💩💩💩💩 SIGMA IS HERE 💩💩💩💩');
troll.onerror = () => console.log(`💩💩💩💩 pooping rn 💩💩💩💩`);

function isLossPointHidden(x, y) {
	return depth > 70 && (x + y % 3) % 3 !== 0
}

function drawHud(ctx) {
	const canvas = ctx.canvas;
	const scale = Math.max(canvas.width / 1366, canvas.height / 768);

	ctx.clearRect(0, 0, canvas.width, canvas.height);

	const W = canvas.width / scale;
	const H = canvas.height / scale;

	ctx.save();
	ctx.scale2(scale);

	ctx.lineCap = ctx.lineJoin = 'round';
	ctx.font = 'normal 10px monospace';
	ctx.textBaseline = 'middle';	
	ctx.textAlign = 'center';

	if (showingLossLandscape) {
		ctx.textBaseline = 'bottom';

		const unlockedColor = getHoverColor();

		if (unlockedPointList.length > 1) {
			ctx.beginPath();
			for (let i = 0; i < unlockedPointList.length - 1; i++) {
				const a = project2(...unlockedPointList[i]);
				const b = project2(...unlockedPointList[i + 1]);
				if (a && b) {
					ctx.moveTo(a[0] * W, a[1] * H);
					ctx.lineTo(b[0] * W, b[1] * H);
				}
			}
			ctx.setLineDash([7, 10]);
			ctx.lineDashOffset = -now / 10 % 1000;
			ctx.strokeStyle = '#fff';
			ctx.lineWidth = 2;
			ctx.stroke();
			ctx.setLineDash([]);
		}

		for (let i = 0; i < lossLandscape.points.length; i++) {
			if (!(i in unlockedPoints) && isLossPointHidden(i % lossLandscapeLength, Math.floor(i / lossLandscapeLength))) continue;

			const loss = lossLandscapePoints[i];
			const p = project2(...lossLandscape.points[i]);
			if (p) {
				ctx.fillStyle = i in unlockedPoints ? unlockedColor : colors.activation;
				ctx.fillText(loss.toFixed(2), p[0] * W, p[1] * H);
			}
		}

		const p = depth < 4.666 && project2(0.5, -75, 2);
		if (p) {
			ctx.save();
			ctx.translate(p[0] * W, p[1] * H);
			ctx.rotate(now / 466.6 % PI2);
			ctx.font = 'bolder 16.66px monospace';
			ctx.fillStyle = 'brown';
			ctx.textBaseline = 'middle';
			ctx.fillText('f💩ggit', 0, 0);
			ctx.restore();
		}

		if (hoveredPoint) block: {
			const p = project2(...hoveredPoint.pos);
			if (!p) break block;

			const texts = [
				`(${(hoveredPoint.x.toFixed(2))}, ${hoveredPoint.y.toFixed(2)})`, 
				`Loss: ${hoveredPoint.loss.toFixed(2)}`
			];

			const t = unlockedPoints[hoveredPoint.i];
			t && texts.push(`txtJus4u: ${t}`);
			
			const widths = [];

			const fontSize = 10;
			const padding = 5;
			const gap = 3;

			let width = 0;
			let height = padding * 2 + texts.length * (fontSize + gap) - gap;

			ctx.font = `normal ${fontSize}px monospace`;

			for (let i = 0; i < texts.length; i++) {
				const w = ctx.measureText(texts[i]).width;
				if (w > width) width = w;
				widths.push(w);
			}

			width += padding * 2;

			ctx.save();
			ctx.translate(p[0] * W, p[1] * H - (isLossPointHidden(hoveredPoint.ix, hoveredPoint.iy) ? 14 : 22) + 6 * (1 - hoverT));
			ctx.scale(0.4 + 0.6 * hoverT, 1)
			ctx.translate(-width / 2, -height);

			ctx.globalAlpha *= hoverT;

			ctx.beginPath();
			ctx.roundRect(0, 0, width, height, 3);
			ctx.save();
			ctx.translate(width / 2, height);
			ctx.moveTo(-6, 0);
			ctx.lineTo(0, 6 * hoverT);
			ctx.lineTo(6, 0);
			ctx.closePath();
			ctx.restore();
			ctx.fillStyle = t ? unlockedColor : 'rgba(0, 0, 0, 0.3)';
			ctx.fill();

			if (hoveredPoint.t > 0) {
				ctx.save();
				ctx.clip();
				ctx.fillStyle = unlockedColor;
				ctx.fillRect(0, 0, width, height * hoveredPoint.t);
				ctx.restore();
			}

			ctx.translate(width / 2, padding);

			ctx.textBaseline = 'top';
			ctx.textAlign = 'center';
			ctx.fillStyle = '#fff';

			for (const text of texts) {
				ctx.fillText(text, 0, 0);
				ctx.translate(0, fontSize + gap);
			}

			ctx.restore();
		}
	} else {
		drawNetworkHud(ctx, W, H);
	}

	// cosmetics

	if (666 && Math.hypot(cx - ropePos[0], cy - ropePos[1], cz - ropePos[2]) < 100) block: {
		const p = project2(ropePos[0], ropePos[1] - 12, ropePos[2]);
		if (!p) break block;

		ctx.save();
		ctx.translate(p[0] * W, p[1] * H);
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.font = 'bolder 50px monospace';
		ctx.fillStyle = 'brown';
		ctx.fillText('भालट', 0, 4);

		ctx.beginPath();
		ctx.moveTo(-50, -20);
		ctx.lineTo(50, 20);
		ctx.moveTo(50, -20);
		ctx.lineTo(-50, 20);
		ctx.strokeStyle = 'red';
		ctx.lineWidth = 5;
		ctx.stroke();

		ctx.font = 'normal 10px monospace';
		ctx.fillStyle = getHoverColor();
		ctx.fillText('⛧6⛧6⛧6⛧', 0, 0);

		ctx.fillStyle = 'white';
		ctx.textBaseline = 'top';
		ctx.fillText(`▲चूहा: मुझे लंडी के साथ सम्भोग दो 3======>`, 0, 28);
		ctx.fillText('फ: लंडी लंडी लंडी बड़ी लंडी बनो XD XD', 0, 28 + 13);

		ctx.restore();
	}

	const p = project2(...lightPos);
	if (p) block: {
		const x = p[0] * W;
		const y = p[1] * H;

		ctx.save();
		ctx.translate(x, y);
		ctx.rotate((now / 1000) % PI2);

		const t = Math.sin(now / 200) * 0.5 + 0.5;

		ctx.beginPath();
		ctx.arc(0, 0, 10, 0, PI2);
		ctx.strokeStyle = ctx.fillStyle = '#fff';
		ctx.globalAlpha = 0.1;
		ctx.fill();
		ctx.globalAlpha = 1;
		ctx.stroke();

		ctx.beginPath();

		const n = 10;
		const a = PI2 / n;
		for (let i = 0; i < n; i++) {
			ctx.rotate(a);
			ctx.moveTo(20, 0);
			ctx.lineTo(27 + (i % 2 === 0 ? 1 - t : t) * 5, 0);
		}
		ctx.lineCap = 'round';
		ctx.lineWidth = 2;
		ctx.stroke();

		ctx.restore();

		if (troll.width > 0 && Math.hypot(x * scale - pointerX, y * scale - pointerY) < 10 * scale) {
			ctx.save();
			ctx.translate(W / 2, H / 2);

			ctx.beginPath();
			ctx.moveTo(2.666, 100);
			ctx.lineTo(2.666, 2666 * 0.2666);
			ctx.lineWidth = 5;
			ctx.fillStyle = ctx.strokeStyle = `#f9f9f9`;
			ctx.stroke();

			ctx.filter = 'grayscale(1) invert(1)';
			ctx.drawImage(troll, -100, -100, 200, 200);
			ctx.filter = 'none';
			ctx.textBaseline = 'bottom';
			ctx.textAlign = 'center';
			ctx.font = `normal 12px monospace`;
			
			ctx.fillText(iCodedThisFunctionLOLXDIAmLandi() + ' v hav Dcided ur fate :p ' + iCodedThisFunctionLOLXDIAmLandi(), 0, -115);

			ctx.restore();
		}
	}

	// graph

	ctx.save();
	ctx.translate(12, H - 12 - 16);

	const graphWidth = 90;
	const graphHeight = 50;

	for (const key in graphs) {
		const graph = graphs[key];
		if (!graph.visible) continue;

		ctx.save();
		showT < 1 && ctx.translate(0, (1 - Math.pow(showT, graph.i)) * graphHeight * 4);

		ctx.beginPath();
		let y = 0;
		if (graph.points.length === 0) {
			y = -graphHeight;
			ctx.lineTo(0, y);
		} else {
			const l = Math.max(1, graph.points.length - 1);
			for (let i = 0; i < graph.points.length; i++) {
				const v = graph.points[i];
				const x = i / l * graphWidth;
				y = -v / graph.max * graphHeight;
				ctx.lineTo(x, y);
			}
		}
		ctx.lineTo(graphWidth, y);
		ctx.lineTo(graphWidth, 0);
		ctx.lineTo(0, 0);
		ctx.closePath();
		ctx.fillStyle = '#333';
		ctx.globalAlpha = 0.3;
		ctx.fill();
		ctx.lineWidth = 1;
		ctx.strokeStyle = '#888';
		ctx.globalAlpha = 1;
		ctx.stroke();

		let n = graph.points[graph.points.length - 1] || 0;

		const matrix = ctx.getTransform();
		const a = new DOMPoint(0, -graphHeight).matrixTransform(matrix);
		const b = new DOMPoint(graphWidth, 0).matrixTransform(matrix);

		if (pointerX > a.x && pointerX < b.x && pointerY > a.y && pointerY < b.y) {
			const f = (pointerX - a.x) / (b.x - a.x);
			const x = f * graphWidth;

			if (graph.points.length > 0) {
				const i = Math.round(f * (graph.points.length - 1));
				n = graph.points[i];
			}

			ctx.save();
			ctx.clip();

			ctx.beginPath();
			ctx.moveTo(x, 0);
			ctx.lineTo(x, -graphHeight);
			ctx.stroke();

			ctx.restore();
		}

		ctx.fillStyle = '#888';
		ctx.font = 'normal 16px monospace';
		ctx.textBaseline = 'bottom';
		ctx.textAlign = 'right';
		ctx.fillText(graph.formatter ? graph.formatter(n) : n.toFixed(2), graphWidth, 0);

		ctx.fillStyle = '#fff';
		ctx.font = 'normal 10px monospace';
		ctx.textBaseline = 'top';
		ctx.textAlign = 'left';
		ctx.fillText(graph.name, 0, 7);

		ctx.restore();

		ctx.translate(graphWidth + 15, 0);
	}

	ctx.restore();

	// progress

	ctx.save();
	ctx.translate(-400 * (1 - showT), H - 130);

	ctx.beginPath();
	ctx.rect(-5, -18, 250 + 5, 36);
	ctx.fillStyle = '#333';
	ctx.globalAlpha = 0.3;
	ctx.fill();
	ctx.lineWidth = 1;
	ctx.strokeStyle = '#888';
	ctx.lineCap = ctx.lineJoin = 'round';
	ctx.globalAlpha = 1;
	ctx.stroke();

	ctx.globalAlpha = 0.1;
	ctx.fillStyle = '#fff';
	ctx.fillRect(0, -14, 246 * progress, 28);

	ctx.globalAlpha = 1;
	ctx.fillStyle = '#fff';
	ctx.font = 'normal 10px monospace';
	ctx.textBaseline = 'middle';
	ctx.textAlign = 'left';
	ctx.fillText(progressText + (progress !== 1 ? '.'.repeat((now / 1000 % 1) * 10) : ''), 10, 0);

	ctx.restore();

	// drag

	if (dragT > 0) {
		ctx.save();
		ctx.translate(W / 2, H / 2);
		ctx.scale2(dragT);
		ctx.globalAlpha = dragT;

		ctx.setLineDash([12, 15]);
		ctx.lineDashOffset = now / 10 % 1000;
		ctx.lineWidth = 2;
		ctx.strokeStyle = '#888';
		ctx.lineCap = ctx.lineJoin = 'round';
		ctx.beginPath();
		ctx.roundRect(-250, -100, 500, 200, 20);
		ctx.fillStyle = 'hsla(0, 0%, 100%, 0.1)';
		ctx.fill();
		ctx.stroke();

		ctx.fillStyle = '#fff';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.font = `normal 17px monospace`;
		ctx.fillText('drop to import XD', 0, (1 - dragT) * -40);

		ctx.restore();
	}

	ctx.restore();
}

function drawNetworkHud(ctx, W, H) {
	for (const item of texts) {
		const p = project2(...item.pos);
		if (p) {
			ctx.textBaseline = 'bottom';
			ctx.textAlign = 'center';
			ctx.fillStyle = colors.label;
			ctx.fillText(item.text, p[0] * W, p[1] * H - 4.666);
		}
	}

	const outputLayer = layers[layers.length - 1];

	for (let i = 0; i < outputLayer.outputLength; i++) {
		const bi = outputLayer.offset + i;
		const [bx, by, bz] = boxes[bi];
		const p = project2(bx, by, bz);
		if (p) {
			const b = project(bx + 1, by + 1, bz + 1);
			const offset = 7 + Math.hypot(p[0] - b[0], p[1] - b[1], p[2] - b[2]) * H;

			const [x, y] = p;
			const dir = x > 0.5 ? 1 : -1;
			ctx.save();
			ctx.translate(x * W, y * H);
			ctx.textBaseline = 'middle';
			ctx.textAlign = x > 0.5 ? 'left' : 'right';
			ctx.fillStyle = colors.label;
			ctx.fillText(i, dir * offset, 0);

			if (prediction) {
				const highlight = i === prediction.result;
				const f = lerp(startActivationData[bi], activationData[bi], predictT);
				if (f > 0) {
					const w = f * 69;
					ctx.fillStyle = `hsl(0, 40%, ${f * 20 + 17}%)`;
					ctx.fillRect(-dir * offset + (dir === 1 ? -w : 0), -8, w, 16);
				}

				if (predictT > 0.99 && predictT < 1) {
					ctx.textAlign = x < 0.5 ? 'left' : 'right';
					ctx.fillStyle = colors.activation;
					ctx.fillText(prediction.probs[i].toFixed(2), -dir * (offset + 2), 0);

					if (highlight) {
						ctx.scale(dir, 1);
						ctx.translate(offset + 15 + (Math.sin((now / 200 % 1) * PI2) * 0.5 + 0.5) * 5, -2);
						ctx.beginPath();
						ctx.moveTo(0, 0);
						ctx.lineTo(15, 12);
						ctx.lineTo(15, 5);
						ctx.lineTo(35, 5);
						ctx.lineTo(35, -5);
						ctx.lineTo(15, -5);
						ctx.lineTo(15, -12);
						ctx.closePath();
						ctx.fillStyle = getHoverColor();
						ctx.fill();
					}
				}
			}

			ctx.restore();
		}
	}

	let kernelN = Date.now() / 555;
	const t = Math.min(1, (kernelN % 1) / 0.3);
	kernelN = Math.floor(kernelN);

	for (let i = 1; i < layers.length; i++) {
		const layer = layers[i];
		const ks = layer.kernelSize;
		if (!ks) continue;
		
		let prevLayer = layers[i - 1];
		if (prevLayer.type === 'Activation') prevLayer = layers[i - 2];

		const outputSize = layer.outputSize;
		const inputSize = prevLayer.outputSize;
		
		let nx = kernelN % outputSize;
		let ny = Math.floor(kernelN / outputSize) % outputSize;

		if (nx < outputSize - 1) {
			nx += t;
			if (ny % 2 === 1) nx = outputSize - 1 - nx;
		} else if (ny < outputSize - 1) {
			if (ny % 2 === 1) nx = outputSize - 1 - nx;
			ny += t;
		}
		
		let sx = nx;
		let sy = ny;
		let ex = sx;
		let ey = sy;
		if (layer.type === 'MaxPool') {
			sx *= ks;
			sy *= ks;
			ex = sx + ks - 1;
			ey = sy + ks - 1;
		} else if (layer.type === 'Conv') {
			ex += ks - 1;
			ey += ks - 1;
		}

		const kernelZ = boxes[layer.offset - 1][2] + 0.5;

		const kernelPoints = projectPoints([
			[getCoord(sx, inputSize) - 0.5, -getCoord(sy, inputSize) + 0.5, kernelZ], 
			[getCoord(ex, inputSize) + 0.5, -getCoord(sy, inputSize) + 0.5, kernelZ], 
			[getCoord(ex, inputSize) + 0.5, -getCoord(ey, inputSize) - 0.5, kernelZ], 
			[getCoord(sx, inputSize) - 0.5, -getCoord(ey, inputSize) - 0.5, kernelZ]
		]);

		ctx.beginPath();
		if (kernelPoints) {
			for (const [x, y] of kernelPoints) {
				ctx.lineTo(x * W, y * H);
			}
			ctx.lineTo(kernelPoints[0][0] * W, kernelPoints[0][1] * H);

			const list = [
				[0, 1, 3, 2], 
				[0, 3, 1, 2]
			];

			for (let [a, b, c, d] of list) {
				a = kernelPoints[a];
				b = kernelPoints[b];
				c = kernelPoints[c];
				d = kernelPoints[d];

				for (let i = 1; i < ks; i++) {
					const f = i / ks;
					ctx.moveTo(lerp(a[0], b[0], f) * W, lerp(a[1], b[1], f) * H);
					ctx.lineTo(lerp(c[0], d[0], f) * W, lerp(c[1], d[1], f) * H);
				}
			}
		}

		const neuronX = getCoord(nx, outputSize);
		const neuronY = -getCoord(ny, outputSize);
		const neuronZ = layer.z - 0.5;

		const neuronPoints = projectPoints([
			[neuronX - 0.5, neuronY + 0.5, neuronZ], 
			[neuronX + 0.5, neuronY + 0.5, neuronZ], 
			[neuronX + 0.5, neuronY - 0.5, neuronZ], 
			[neuronX - 0.5, neuronY - 0.5, neuronZ]
		]);

		if (neuronPoints) {
			ctx.moveTo(neuronPoints[0][0] * W, neuronPoints[0][1] * H);
			for (const [x, y] of neuronPoints) {
				ctx.lineTo(x * W, y * H);
			}
			ctx.lineTo(neuronPoints[0][0] * W, neuronPoints[0][1] * H);

			if (kernelPoints) {
				for (let i = 0; i < kernelPoints.length; i++) {
					ctx.moveTo(kernelPoints[i][0] * W, kernelPoints[i][1] * H);
					ctx.lineTo(neuronPoints[i][0] * W, neuronPoints[i][1] * H);
				}
			}
		}

		ctx.lineWidth = 2;
		ctx.strokeStyle = `#999`;
		ctx.stroke();
	}

	if (picked) block: {
		const points = projectPoints(picked.points);
		if (!points) break block;

		ctx.save();
		ctx.beginPath();
		for (const [x, y] of points) {
			ctx.lineTo(x * W, y * H);
		}
		ctx.closePath();

		ctx.fillStyle = `#fff`;
		const t = (Math.sin(now / 100) * 0.5 + 0.5);
		ctx.globalAlpha = t * 0.069 + 0.1;
		ctx.fill();
		ctx.globalAlpha = 1;
		ctx.lineWidth = 2;
		ctx.strokeStyle = '#fff';
		ctx.stroke();

		ctx.save();
		ctx.clip();

		const z = picked.points[0][2];
		const r = Math.abs(picked.points[0][0]) * 0.69969;

		ctx.beginPath()
		for (let i = 0; i < 5; i++) {
			const a = PI2 * i / 2.5 + now / 2666 + picked.i / picked.layer.depth * 2.666;
			const [x, y] = project(Math.cos(a) * r, Math.sin(a) * r, z);
			ctx.lineTo(x * W, y * H);
		}
		ctx.closePath();
		ctx.globalAlpha = 0.0555 + t * 0.0666;
		ctx.lineWidth = 8.555;
		ctx.stroke();

		ctx.restore();

		const size = 90;
		ctx.translate(picked.x * W, picked.y * H - 10 - size);

		if (picked.image) {
			ctx.imageSmoothingEnabled = false;
			ctx.drawImage(picked.image, -size / 2, 0, size, size);
		}

		ctx.translate(0, -5);
		ctx.fillStyle = colors.label;
		ctx.textAlign = 'center';
		ctx.textBaseline = 'bottom';
		ctx.fillText(picked.name, 0, 0);

		ctx.restore();
	}
}

function projectPoints(points) {
	const result = [];
	for (const point of points) {
		const p = project(...point);
		if (p[2] < 0 || p[2] > 1) return;
		result.push(p);
	}
	return result;
}

let pointerX = 0;
let pointerY = 0;
document.onmousemove = function (event) {
	pointerX = event.clientX / window.innerWidth * hudCanvas.width;
	pointerY = event.clientY / window.innerHeight * hudCanvas.height;
}

let nrx = 0.1;
let nry = -7;
let nDepth = 100;

let rx = 0.6;
let ry = -0.5;
let depth = 180;

let cx = 0;
let cy = 0;
let cz = 0;

let ncx = 0;
let ncy = 0;
let ncz = 0;

const minDepth = 2;
const maxDepth = 130;

canvas.onwheel = function (event) {
	zoom(event.deltaY > 0 ? 1.1 : 0.9);
}

canvas.oncontextmenu = () => false;

function zoom(f) {
	nDepth *= f;
	nDepth = Math.max(minDepth, Math.min(nDepth, maxDepth));
}

function onPointerMove(pointer, pan) {
	if (lastPointer) {
		let dx = -(pointer[0] - lastPointer[0]);
		let dy = pointer[1] - lastPointer[1];
		
		if (pan) {
			dx *= 0.3;
			dy *= 0.3;
			ncx += viewMatrix[0] * dx + viewMatrix[1] * dy;
			ncy += viewMatrix[4] * dx + viewMatrix[5] * dy;
			ncz += viewMatrix[8] * dx + viewMatrix[9] * dy;
		} else {
			nrx += dy * 0.01;
			nry += dx * 0.01;
			nrx = Math.max(-Math.PI / 2, Math.min(nrx, Math.PI / 2));
		}

		lastPointer = pointer;
	}
}

let picked;
let hoveredPoint;
let hoverT = 0;

let lastPointer;
canvas.onmousedown = function (event) {
	if (event.button === 0) {
		lastPointer = [event.clientX, event.clientY];
	}
}
window.onmousemove = function (event) {
	const pointer = [event.clientX, event.clientY];
	onPointerMove(pointer, event.shiftKey || event.ctrlKey);	

	picked = null;

	for (const layer of layers) {
		if (layer.type !== 'Conv') continue;

		let minZ = Infinity;

		for (let i = 0; i < layer.depth; i++) {
			const s = layer.outputSize * gap + 0.5;
			const z = layer.z + i * depthGap;

			const points = [
				[-s, s, z], 
				[s, s, z], 
				[s, -s, z], 
				[-s, -s, z]
			];

			let midZ = 0;

			let inView = true;

			const path = new Path2D();
			for (const point of points) {
				const p = project(...point);
				if (p[2] < 0 || p[2] > 1) {
					inView = false;
					break;
				}
				const [x, y, z] = p;
				path.lineTo(x * window.innerWidth, y * window.innerHeight);
				midZ += z;
			}

			if (!inView) continue;

			midZ /= points.length;

			if (hudCtx.isPointInPath(path, ...pointer) && midZ < minZ) {
				picked = {
					layer, 
					i, 
					points
				};
				minZ = midZ;
			}
		}
	}

	if (picked) {
		picked.x = pointer[0] / window.innerWidth;
		picked.y = pointer[1] / window.innerHeight;

		const layer = picked.layer;
		picked.name = layer.type + ' Kernel #' + (picked.i + 1);

		if (layer.kernels) {
			const l = layer.kernelSize * layer.kernelSize;

			picked.image = createImage(
				layer.kernels.slice(picked.i * l, picked.i * l + l), 
				layer.kernelSize
			);
		}
	}

	if (showingLossLandscape) {
		const old = hoveredPoint;
		hoveredPoint = null;

		let minDistance = Infinity;
		for (let i = 0; i < lossLandscape.points.length; i++) {
			const point = lossLandscape.points[i];
			const p = project2(...point);
			if (p && Math.hypot(p[0] * window.innerWidth - pointer[0], p[1] * window.innerHeight - pointer[1]) < 10 && p[2] < minDistance) {
				const ix = i % lossLandscapeLength;
				const iy = Math.floor(i / lossLandscapeLength);

				hoveredPoint = {
					i, 
					ix, 
					iy, 
					x: toRange(lossLandscapeRange, ix / lossLandscapeLength), 
					y: toRange(lossLandscapeRange, iy / lossLandscapeLength), 
					pos: point, 
					loss: lossLandscapePoints[i]
				};
				if (!lastPointer) hoveredPoint.t = 0;
				minDistance = p[2];
			}
		}

		if (old?.pos !== hoveredPoint?.pos) hoverT = 0;
	}
}
window.onmouseup = function (event) {
	if (event.button === 0) {
		lastPointer = null;
	}
}

const onTouchMove = {};
const onTouchEnd = {};

window.ontouchstart = function (event) {
	for (const touch of event.changedTouches) {
		touch.target.onTouchStart?.(touch, event);
	}
}

window.addEventListener('touchmove', function (event) {
	for (const touch of event.changedTouches) {
		const handler = onTouchMove[touch.identifier];
		if (handler) {
			event.preventDefault();
			handler(touch, event);
		}
	}
}, { passive: false });

window.ontouchend = function (event) {
	for (const touch of event.changedTouches) {
		const id = touch.identifier;
		onTouchEnd[id]?.(touch, event);
		delete onTouchEnd[id];
		delete onTouchMove[id];
	}
}

const touches = [];
let lastTouchDistance = 0;

function getTouchPointer() {
	let x = 0;
	let y = 0;
	for (const touch of touches) {
		x += touch.x;
		y += touch.y;
	}
	return [x / touches.length, y / touches.length];
}

function getTouchDistance() {
	if (touches.length < 2) return 0;
	const [a, b] = touches;
	return Math.hypot(a.x - b.x, a.y - b.y);
}

canvas.onTouchStart = function (touch) {
	if (touches.length < 2) {
		const t = {
			id: touch.identifier, 
			x: touch.clientX, 
			y: touch.clientY
		};
		touches.push(t);
		lastPointer = getTouchPointer();
		lastTouchDistance = getTouchDistance();

		onTouchMove[t.id] = function (touch) {
			t.x = touch.clientX;
			t.y = touch.clientY;
			onPointerMove(getTouchPointer(), touches.length >= 2);

			const d = getTouchDistance();
			if (d) {
				zoom(Math.pow(lastTouchDistance / d, 1.666));
				lastTouchDistance = d;
			}
		}

		onTouchEnd[t.id] = function (touch) {
			const i = touches.findIndex(t => t.id === touch.identifier);
			if (i > -1) {
				touches.splice(i, 1);
				lastPointer = touches.length > 0 ? getTouchPointer() : null;
			}
		}
	}
}

function createImage(data, size) {
	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext('2d');

	const imageData = ctx.createImageData(size, size);

	let min = Infinity;
	let max = -Infinity;
	for (let i = 0; i < data.length; i++) {
		min = Math.min(data[i], min);
		max = Math.max(data[i], max);
	}

	for (let i = 0; i < data.length; i++) {
		const f = (data[i] - min) / (max - min) * 255;
		imageData.data.set([f, f, f, 255], i * 4);
	}

	ctx.putImageData(imageData, 0, 0);

	return canvas;
}

const ropePos = [0, 400, -450];
const rope = createRope();

function createRope() {
	const points = [];

	const size = 5.55;
	const [X, Y, Z] = ropePos;
	const sy = 2.666;
	const h = 666;

	const n = 69;
	for (let i = 0; i <= n; i++) {
		const f = i / n;
		const a = f * PI2 - Math.PI / 2;
		let x = Math.sin(a);
		x = f < 0.5 ? Math.pow(x, 5) : x;
		points.push([
			X + x * size, 
			Y + Math.cos(a) * (f < 0.5 ? sy : 1) * size
		]);
	}

	const positions = [
		X, Y + size * sy, Z, 
		X, Y + h, Z
	];

	const code = '.-. .- -- '.split('');
	
	let sum = 0;
	for (let i = 0; i < code.length; i++) {
		const c = code[i];
		code[i] = [c, sum];
		sum += c === ' ' ? 3 : 1;
	}

	for (const [c, v] of code) {
		if (c === ' ') continue;
		const a = v / sum * PI2;
		const d = c === '-' ? 36 : 14;
		positions.push(
			X, Y + h, Z, 
			X + Math.sin(a) * d, Y + h, Z + Math.cos(a) * d
		);
	}

	for (let i = 0; i < points.length; i++) {
		positions.push(
			...points[i], Z, 
			...points[(i + 1) % points.length], Z
		);
	}

	return {
		buffer: createBuffer(new Float32Array(positions)),
		count: positions.length / 3
	};
}

const browseUI = new BrowseUI();

let showT = 0;
let shakeT = 0;

let projectionMatrix, viewMatrix;

let showingLossLandscape = false;

let unlockT = 1;

const unlockedPoints = {};
const unlockedPointList = [];

const landscapeTexts = DEV ? ['XD', 'testing XD', 'LOL'] : [
	'you died', 
	'very good', 
	'666', 
	'new stuff', 
	'keep going', 
	'stop!', 
	'repeat', 
	'try again', 
	'▲', 
	'rip lol', 
	'bye bye', 
	'nope', 
	'yep', 
	'never return XD', 
	'555', 
	'2 coins', 
	'green', 
	'red', 
	'fuck',
	'fish',
	'rope4u'
];
const unlockTime = DEV ? 0.4666 : 4.666;

let now = 0;
let lastTime = Date.now();
let dt = 0;
let dts = 0;

function update() {
	now = Date.now();
	dt = now - lastTime;
	dts = dt / 1000;
	lastTime = now;

	if (!lastPointer) {
		nry += 0.015 * settings.orbitSpeed * dts;
	}

	let lf = getLerpFactor(0.05);
	rx = lerpAngle(rx, nrx, lf);
	ry = lerpAngle(ry, nry, lf);
	depth = lerp(depth, nDepth, lf);

	cx = lerp(cx, ncx, lf);
	cy = lerp(cy, ncy, lf);
	cz = lerp(cz, ncz, lf);

	showingLossLandscape = settings.lossLandscape && lossLandscape && !isTraining();
	graphs.lossCurve.visible = showingLossLandscape;

	lf = getLerpFactor(0.1);
	showT = lerp(showT, 1, lf);
	headerEl.style.transform = `translateX(${(1 - showT) * 200}%)`;
	settingsEl.style.transform = `translateY(${(1 - showT) * -200}%)`;
	sketchEl.style.transform = `translateY(${(1 - showT) * 200}%)`;

	if (hoveredPoint?.t !== undefined && !(hoveredPoint.i in unlockedPoints)) {
		if (lastPointer) {
			hoveredPoint.t += dts / unlockTime;
			if (hoveredPoint.t > 1) {
				unlockedPoints[hoveredPoint.i] = landscapeTexts[Math.floor(Math.random() * landscapeTexts.length)];
				unlockedPointList.push(lossLandscape.points[hoveredPoint.i]);
				unlockT = 0;
				hoveredPoint.t = 0;

				gl.useProgram(planeProgram);
				gl.uniform3fv(planeProgram.uniforms.center, hoveredPoint.pos);
			}
		} else {
			hoveredPoint.t = 0;
		}
	}

	unlockT += dts / 1;
	unlockT > 1 && (unlockT = 1);

	hoverT = lerp(hoverT, 1, getLerpFactor(0.2));
	dragT = lerp(dragT, dragging ? 1 : 0, getLerpFactor(0.2));
	shakeT = lerp(shakeT, Math.abs(cx) > 2500 || Math.abs(cy) > 2500 || Math.abs(cz) > 2500 ? 1 : 0, getLerpFactor(0.1));

	browseUI.update();

	graphs.lossCurve.visible = settings.lossLandscape;

	if (loaded) {
		// prevent multiple msgs in chk import XD
		if (datasetNeedsUpdate) {
			worker.postMessage({
				id: 'createDatasets', 
				trainSplit: settings.trainSplit, 
				dataSplit: settings.dataSplit
			});
			datasetNeedsUpdate = false;
		}

		if (!showingLossLandscape && !predicting && (userInput || canAutoPredict && predictT === 1)) {
			predicting = true;
			worker.postMessage({
				id: 'predict', 
				x: userInput
			});
			userInput = null;
		}

		if (isTraining()) {
			if (!epoching) {
				epoching = true;
				worker.postMessage({
					id: 'train'
				});
			}
		} else if (settings.lossLandscape) {
			if (!lossCurving && graphs.lossCurve.points.length <= lossCurveLength) {
				lossCurving = true;
				worker.postMessage({
					id: 'lossCurve', 
					x: toRange(lossCurveRange, graphs.lossCurve.points.length / lossCurveLength)
				});
			}

			if (!lossLandscaping && lossLandscapePoints.length < lossLandscapeSize) {
				lossLandscaping = true;
				const i = lossLandscapePoints.length;
				worker.postMessage({
					id: 'lossLandscape', 
					x: toRange(lossLandscapeRange, (i % lossLandscapeLength) / lossLandscapeLength), 
					y: toRange(lossLandscapeRange, Math.floor(i / lossLandscapeLength) / lossLandscapeLength)
				});
			}
		}

		predictT = lerp(predictT, 1, getLerpFactor(0.1));
	}
}

function render() {
	let rotX = rx;
	let rotY = ry;

	if (shakeT > 0) {
		rotX += Math.sin(now / 150) * 0.1 * shakeT;
		rotY += Math.sin(now / 300) * Math.cos(now / 120) * 0.15 * shakeT;
	}

	const cosX = Math.cos(rotX);
	const sinX = Math.sin(rotX);
	const cosY = Math.cos(rotY);
	const sinY = Math.sin(rotY);

	viewMatrix = [
		cosY, sinY * -sinX, sinY * cosX, 0, 
		0, cosX, sinX, 0, 
		-sinY, cosY * -sinX, cosY * cosX, 0, 
		0, 0, -depth, 1
	];

	viewMatrix[12] -= viewMatrix[0] * cx + viewMatrix[4] * cy + viewMatrix[8] * cz;
	viewMatrix[13] -= viewMatrix[1] * cx + viewMatrix[5] * cy + viewMatrix[9] * cz;
	viewMatrix[14] -= viewMatrix[2] * cx + viewMatrix[6] * cy + viewMatrix[10] * cz;

	const near = 0.1;
	const far = 1000;
	const fov = 60;
	const f = 1 / Math.tan(fov * Math.PI / 360);
	const nf = 1 / (near - far);
	const aspect = canvas.width / canvas.height;

	projectionMatrix = [
		f / aspect, 0, 0, 0, 
		0, f, 0, 0, 
		0, 0, (near + far) * nf, -1, 
		0, 0, 2 * far * near * nf, 1
	];

	drawHud(hudCtx);

	gl.viewport(0, 0, canvas.width, canvas.height);

	const chai = `-. .- .-. . -. -.. .-. / -.. .- .- -- --- -.. .- .-. .- -.. .- .- ... / -- --- -.. . . / -.- .- / .--- .- .-.. -.. / .... . . / -. .. -.. .... .- -. / .... --- .-.-.- / -- --- -.. . . / -... .- .... ..- - / -... --- --- -.. .... . / .... .- .. -. .-.-.- / -... ..- -.. .... .- .- .--. . / ... . / -- .-. - -.-- ..- .-.-.-`;
	const char666 = chai[Math.floor((now / 76e3 % 1) * chai.length)];
	const r = {
		'-': 0.8, 
		'.': 0, 
		' ': 0.1, 
		'/': 0.25
	}[char666] || Math.random();
	gl.clearColor(r, r, r, 1);
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

	renderBg();

	renderLines(rope.buffer, rope.count, [0.27, 0.18, 0.03, 1]);
	showingLossLandscape ? renderLossLandscape() : renderBoxes();
}

function renderBg() {
	gl.useProgram(bgProgram);
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, map);
	gl.uniform1i(bgProgram.uniforms.map, 0);

	gl.uniformMatrix4fv(bgProgram.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(bgProgram.uniforms.viewMatrix, false, viewMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
	gl.enableVertexAttribArray(bgProgram.attributes.position);
	gl.vertexAttribPointer(bgProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);

	gl.cullFace(gl.FRONT);
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
	gl.drawElements(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0);			
	gl.cullFace(gl.BACK);

	gl.disableVertexAttribArray(bgProgram.attributes.position);

	gl.clear(gl.DEPTH_BUFFER_BIT);
}

function renderBoxes() {
	gl.useProgram(program);

	gl.uniform1f(program.uniforms.t, predictT);
	gl.uniform3fv(program.uniforms.lightPos, lightPos);
	gl.uniformMatrix4fv(program.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(program.uniforms.viewMatrix, false, viewMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
	gl.enableVertexAttribArray(program.attributes.position);
	gl.vertexAttribPointer(program.attributes.position, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.position, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
	gl.enableVertexAttribArray(program.attributes.normal);
	gl.vertexAttribPointer(program.attributes.normal, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.normal, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, worldPosBuffer);
	gl.enableVertexAttribArray(program.attributes.worldPos);
	gl.vertexAttribPointer(program.attributes.worldPos, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.worldPos, 1);

	gl.bindBuffer(gl.ARRAY_BUFFER, startActivationBuffer);
	gl.enableVertexAttribArray(program.attributes.startActivation);
	gl.vertexAttribPointer(program.attributes.startActivation, 1, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.startActivation, 1);

	gl.bindBuffer(gl.ARRAY_BUFFER, activationBuffer);
	gl.enableVertexAttribArray(program.attributes.activation);
	gl.vertexAttribPointer(program.attributes.activation, 1, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.activation, 1);

	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
	ext.drawElementsInstancedANGLE(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_SHORT, 0, objectCount);

	gl.disableVertexAttribArray(program.attributes.position);
	gl.disableVertexAttribArray(program.attributes.normal);
	gl.disableVertexAttribArray(program.attributes.worldPos);
	gl.disableVertexAttribArray(program.attributes.activation);
	gl.disableVertexAttribArray(program.attributes.startActivation);
	ext.vertexAttribDivisorANGLE(program.attributes.worldPos, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.startActivation, 0);
	ext.vertexAttribDivisorANGLE(program.attributes.activation, 0);

	// lines

	gl.useProgram(lineProgram);

	gl.uniformMatrix4fv(lineProgram.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(lineProgram.uniforms.viewMatrix, false, viewMatrix);

	gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
	gl.enableVertexAttribArray(lineProgram.attributes.position);
	gl.vertexAttribPointer(lineProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(lineProgram.attributes.position, 0);

	gl.bindBuffer(gl.ARRAY_BUFFER, worldPosBuffer);
	gl.enableVertexAttribArray(lineProgram.attributes.worldPos);
	gl.vertexAttribPointer(lineProgram.attributes.worldPos, 3, gl.FLOAT, false, 0, 0);
	ext.vertexAttribDivisorANGLE(lineProgram.attributes.worldPos, 1);

	ext.drawArraysInstancedANGLE(gl.LINES, 0, mesh.lines.length / 3, objectCount);

	gl.disableVertexAttribArray(lineProgram.attributes.position);
	gl.disableVertexAttribArray(lineProgram.attributes.worldPos);
	ext.vertexAttribDivisorANGLE(lineProgram.attributes.worldPos, 0);
}

function renderLossLandscape() {
	if (lossLandscape.vertexCount > 0) {
		gl.useProgram(planeProgram);

		gl.uniform3fv(planeProgram.uniforms.lightPos, lightPos);
		gl.uniform1f(planeProgram.uniforms.t, Math.sin(unlockT * Math.PI));
		gl.uniformMatrix4fv(planeProgram.uniforms.projectionMatrix, false, projectionMatrix);
		gl.uniformMatrix4fv(planeProgram.uniforms.viewMatrix, false, viewMatrix);

		gl.bindBuffer(gl.ARRAY_BUFFER, lossLandscape.posBuffer);
		gl.enableVertexAttribArray(planeProgram.attributes.position);
		gl.vertexAttribPointer(planeProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);

		gl.bindBuffer(gl.ARRAY_BUFFER, lossLandscape.vertexNormalBuffer);
		gl.enableVertexAttribArray(planeProgram.attributes.normal);
		gl.vertexAttribPointer(planeProgram.attributes.normal, 3, gl.FLOAT, false, 0, 0);

		gl.bindBuffer(gl.ARRAY_BUFFER, lossLandscape.intensityBuffer);
		gl.enableVertexAttribArray(planeProgram.attributes.intensity);
		gl.vertexAttribPointer(planeProgram.attributes.intensity, 1, gl.FLOAT, false, 0, 0);

		gl.disable(gl.CULL_FACE);
		gl.drawArrays(gl.TRIANGLES, 0, lossLandscape.vertexCount);
		gl.enable(gl.CULL_FACE);

		gl.disableVertexAttribArray(planeProgram.attributes.position);
		gl.disableVertexAttribArray(planeProgram.attributes.normal);
		gl.disableVertexAttribArray(planeProgram.attributes.intensity);
	}

	renderLines(lossLandscape.lineBuffer, lossLandscape.lineCount, [0.3, 0.3, 0.3, 0.3]);
}

function renderLines(buffer, count, color) {
	gl.useProgram(planeLineProgram);

	gl.uniformMatrix4fv(planeLineProgram.uniforms.projectionMatrix, false, projectionMatrix);
	gl.uniformMatrix4fv(planeLineProgram.uniforms.viewMatrix, false, viewMatrix);
	gl.uniform4fv(planeLineProgram.uniforms.color, color);

	gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
	gl.enableVertexAttribArray(planeLineProgram.attributes.position);
	gl.vertexAttribPointer(planeLineProgram.attributes.position, 3, gl.FLOAT, false, 0, 0);

	gl.drawArrays(gl.LINES, 0, count);

	gl.disableVertexAttribArray(planeLineProgram.attributes.position);
}

function animate() {
	update();
	render();
	window.requestAnimationFrame(animate);
}

animate();

function createProgram(vert, frag) {
	const vShader = createShader(vert, true);
	const fShader = createShader(frag, false);

	const program = gl.createProgram();
	gl.attachShader(program, vShader);
	gl.attachShader(program, fShader);
	gl.linkProgram(program);

	gl.deleteShader(vShader);
	gl.deleteShader(fShader);

	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		throw new Error(`failed to link program: ${gl.getProgramInfoLog(program)}`);
	}

	const attributeCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
	program.attributes = {};
	for (let i = 0; i < attributeCount; i++) {
		const info = gl.getActiveAttrib(program, i);
		program.attributes[info.name] = gl.getAttribLocation(program, info.name);
	}

	const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
	program.uniforms = {};
	for (let i = 0; i < uniformCount; i++) {
		const info = gl.getActiveUniform(program, i);
		program.uniforms[info.name] = gl.getUniformLocation(program, info.name);
	}

	return program;
}

function createShader(src, isVertex) {
	const shader = gl.createShader(isVertex ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER);
	gl.shaderSource(shader, src);
	gl.compileShader(shader);

	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		throw new Error(`failed to compile ${isVertex ? 'vertex' : 'fragment'} shader! ${gl.getShaderInfoLog(shader)}`);
	}

	return shader;
}

function createBuffer(data, dynamic) {
	const buffer = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
	gl.bufferData(gl.ARRAY_BUFFER, data, dynamic ? gl.DYNAMIC_DRAW :  gl.STATIC_DRAW);
	return buffer;
}

function getHoverColor() {
	return `hsl(${(now / 400 % 1) * 360}deg, 100%, 70%)`;
}

function project(x, y, z) {
	const p = transformVector(transformVector([x, y, z, 1], viewMatrix), projectionMatrix);
	x = p[0] / p[3] * 0.5 + 0.5;
	y = 0.5 - p[1] / p[3] * 0.5;
	z = p[2] / p[3] * 0.5 + 0.5;
	return [x, y, z];
}

function project2(x, y, z) {
	const p = project(x, y, z);
	if (inView(...p)) return p;
}

function transformVector(p, matrix) {
	return [
		p[0] * matrix[0] + p[1] * matrix[4] + p[2] * matrix[8] + p[3] * matrix[12], 
		p[0] * matrix[1] + p[1] * matrix[5] + p[2] * matrix[9] + p[3] * matrix[13], 
		p[0] * matrix[2] + p[1] * matrix[6] + p[2] * matrix[10] + p[3] * matrix[14], 
		p[0] * matrix[3] + p[1] * matrix[7] + p[2] * matrix[11] + p[3] * matrix[15]
	];
}

function inView(x, y, z) {
	return x > 0 && x < 1 && y > 0 && y < 1 && z > 0 && z < 1;
}

function toRange([min, max], f) {
	return min + (max - min) * f;
}

function lerpAngle(a, b, t) {
	let da = (b - a) % PI2;
	da = 2 * da % PI2 - da;
	return a + da * t;
}

function lerp(start, target, t) {
	const d = target - start;
	if (Math.abs(d) < 1e-4) return target;
	return start + d * t;
}

function getLerpFactor(f) {
	return 1 - Math.exp(-f * dt / 16);
}

function BrowseUI() {
	const overlayEl = fromHtml(`<div class="overlay"></div>`);
	uiEl.appendChild(overlayEl);

	const el = fromHtml(`<div class="dialog">
		<div class="dialog-header">dataset explolal</div>
		<div class="dialog-close">[x]</div>
		<div class="wrapper" style="position: relative; flex: 1; margin: 5px;">
			<div class="dialog-content status" style="display: grid; place-content: center;">
				<div>nothing to display xp...</div>
			</div>
		</div>
		<div class="row" style="margin: 5px; margin-top: 0;">
			<div>Digit</div>
			<select class="numbers" style="width: 60px;">${(function () {
				let html = '';
				for (let i = -1; i < 10; i++) {
					html += `<option value="${i}">${i === -1 ? 'All' : i}</option>`;
				}
				return html;
			})()}</select>
			<select class="pages" style="margin-left: auto;"></select>
			<div class="page-count">of 0</div>
			<div class="btn prev-btn disabled">prev</div>
			<div class="btn next-btn disabled">next</div>
		</div>
	</div>`);
	uiEl.appendChild(el);

	el.onmousemove = overlayEl.onmousemove = event => event.stopPropagation();

	const itemsPerPage = 200;
	let filterDigit = -1;

	overlayEl.onclick = el.querySelector('.dialog-close').onclick = function () {
		setVisible(false);
	}

	const wrapperEl = el.querySelector('.wrapper');
	const statusEl = el.querySelector('.status');

	const pageCountEl = el.querySelector('.page-count');
	const pagesEl = el.querySelector('.pages');

	const accuracyEl = el.querySelector('.accuracy');

	const numbersEl = el.querySelector('.numbers');
	numbersEl.value = filterDigit;
	numbersEl.onchange = function () {
		filterDigit = parseInt(this.value);
		reqContent();
	}

	const prevBtnEl = el.querySelector('.prev-btn');
	const nextBtnEl = el.querySelector('.next-btn');

	let pageCount = 0;
	let currPage = 0;

	pagesEl.onchange = function () {
		currPage = parseInt(this.value);
		reqContent();
	}

	nextBtnEl.onclick = function () {
		incPage(1);
	}

	prevBtnEl.onclick = function () {
		incPage(-1);
	}

	function incPage(dir) {
		currPage += dir;
		currPage = Math.max(0, Math.min(currPage, pageCount));
		pagesEl.value = currPage;
		onPageChange();
		reqContent();
	}

	function onPageChange() {
		prevBtnEl.setClass('disabled', currPage <= 0);
		nextBtnEl.setClass('disabled', currPage >= pageCount - 1);
	}

	let lastPage = -1;

	function setContent(data) {
		pageCount = Math.floor(data.totalCount / itemsPerPage);
		currPage = Math.floor(data.start / itemsPerPage);

		statusEl.style.visibility = 'hidden';

		const contentEl = fromHtml(`<div class="dialog-content" style="position: absolute; top: 0; left: 0;"></div>`);
		contentEl.t = contentEl.d = currPage > lastPage ? 1 : -1;

		const lastEl = wrapperEl.lastElementChild
		if (lastEl) {
			lastEl.style.overflow = 'hidden';
			if (lastEl.d !== contentEl.d) {
				lastEl.d *= -1;
			}
		}

		lastPage = currPage;
		wrapperEl.appendChild(contentEl);

		let html = '';
		for (let i = 0; i < pageCount; i++) {
			html += `<option value="${i}">Page ${i + 1}</option>`;
		}
		pagesEl.innerHTML = html;
		pagesEl.value = currPage;
		onPageChange();

		pageCountEl.innerText = `of ${pageCount}`;

		for (let i = 0; i < data.items.length; i++) {
			const item = data.items[i];

			const el = fromHtml(`<div class="preview">
				<div class="tooltip">
					<div>#${item.id}</div>
					<div id="pred-${item.id}"></div>
				</div>
			</div>`);
			el.appendChild(createImage(item.x, inputSize));
			el.item = item;
			el.onclick = onClick;
			el.onmouseenter = onMouseEnter;
			contentEl.appendChild(el);
		}

		if (filterDigit === -1) {
			const text = data.items.map(item => item.y).join('');
			const matches = text.matchAll(/3301|666|1102|2003|2020/g);
			
			for (const match of matches) {
				const l = match[0].length;
				for (let i = 0; i < l; i++) {
					contentEl.children[match.index + i].classList.add('highlight');
				}
			}
		}
	}

	function onMouseEnter() {
		worker.postMessage({
			id: 'elPredict', 
			el: `pred-${this.item.id}`, 
			x: this.item.x
		})
	}

	function onClick() {
		userInput = this.item.x;
		predictT = 1;
	}

	function reqContent() {
		worker.postMessage({
			id: 'dataset', 
			filterDigit, 
			start: currPage * itemsPerPage, 
			count: itemsPerPage
		});
	}

	let visible = false;
	let t = 0;

	function update() {
		const f = getLerpFactor(0.2);
		t = lerp(t, visible ? 1 : 0, f);

		for (let i = wrapperEl.children.length - 1; i >= 0; i--) {
			const el = wrapperEl.children[i];
			if (el.t === undefined) continue;

			const target = wrapperEl.lastElementChild === el ? 0 : -el.d;
			el.t = lerp(el.t, target, f);
			el.style.transform = `translateX(${el.t * 100}%)`;
			if (target && el.t === target) {
				el.remove();
			}
		}

		el.style.transform = `translate(-50%, -50%) scale(${t})`;
		el.style.opacity = overlayEl.style.opacity = t;
		overlayEl.style.display = t === 0 ? 'none' : '';
	}

	function setVisible(v) {
		visible = v;
	}

	this.update = update;
	this.reqContent = reqContent;
	this.setVisible = setVisible;
	this.setContent = setContent;
}

function iCodedThisFunctionLOLXDIAmLandi() {
	return Array.from({ length: 10 }, () => Math.random().toString(32).slice(2, 10)).join('⛧');
}