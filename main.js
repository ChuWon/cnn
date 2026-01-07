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
}

function reset(id) {
	modelId = id || Math.random().toString(32).slice(2);
	epoch = 0;
	lastCheckpointT = 0;
	epoching = false;
	resetGraphs();
}

let epoch = 0;
let epochPercent = 0;
let epoching = false;
let lastCheckpointT = 0;

function onTrain(msg) {
	epoching = false;
	epoch = msg.epoch;
	epochPercent = (msg.epochPercent * 100).toFixed(2);
	progress = Math.min(1, (msg.epoch + msg.epochPercent) / epochs);
	progressText = `training epoch ${epoch}/${epochs}${msg.epochPercent ? ` (${epochPercent}%)` : ''}`;
}

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
			break;

		case 'failed':
			alert(`Failed to load dataset, can't train model. Reload the page to retry.`);
			break;

		case 'epochBatch':
			if (msg.modelId !== modelId) return;
			onTrain(msg);
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
			if (msg.modelId !== modelId) return;
			onTrain(msg);
			addGraph('epochTime', msg.epochTimeTaken);
			addGraph('trainLoss', msg.trainLoss);
			addGraph('trainAccuracy', msg.trainAccuracy);
			addGraph('valLoss', msg.valLoss);
			addGraph('valAccuracy', msg.valAccuracy);
			break;

		case 'prediction': {
			let maxIndex = -1;
			let max = -Infinity;

			for (let i = 0; i < msg.y.length; i++) {
				const p = msg.y[i];
				if (p > max) {
					maxIndex = i;
					max = p;
				}
			}

			console.log(`probs: ${msg.y}\nprediction: ${maxIndex}`);
			alert(`prediction: ${maxIndex}`);
		}	break;

		case 'checkpoint':
			const a = document.createElement('a');
			a.href = URL.createObjectURL(new Blob([msg.json], { type: 'text/plain' }));
			a.download = `cnn-e${epoch}-${epochPercent}%.666`;
			a.click();
			break;

		case 'checkpointData':
			reset(msg.modelId);

			const data = msg.data;

			for (const key in data.settings) {
				const v = data.settings[key];
				if (typeof settings[key] === typeof v) {
					setSetting(key, v);
				}
			}

			for (const key in data.graphs) {
				if (key in graphs) {
					const list = data.graphs[key];
					for (let i = 0; i < list.length; i++) {
						addGraph(key, list[i]);
					}
				}
			}
			break;

		case 'checkpointError':
			alert(`Failed to load checkpoint!\nError: ${msg.error}`);
			break;

		default:
			console.log(`Unknown msg id from worker: ${msg.id}`);
	}
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

function createDatasets() {
	loaded && worker.postMessage({
		id: 'createDatasets', 
		trainSplit: settings.trainSplit, 
		dataSplit: settings.dataSplit
	});
}

function saveCheckpoint() {
	const data = {
		settings, 
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

function predict(image) {
	const size = Math.sqrt(inputLength);

	const canvas = document.createElement('canvas');
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext('2d');

	ctx.drawImage(image, 0, 0, size, size);

	const imageData = ctx.getImageData(0, 0, size, size);

	const x = new Float32Array(inputLength);
	for (let i = 0; i < inputLength; i++) {
		x[i] = imageData.data[i * 4 + 3] / 255;
	}

	worker.postMessage({
		id: 'predict', 
		modelId, 
		x
	});
}

function isTraining() {
	return settings.trainingEnabled && (epoch < epochs || settings.endlessTraining);
}

const epochs = 5;

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
			<div class="btn predict-btn">predict</div>
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

	el.querySelector('.clear-btn').onclick = function () {
		paths.length = 0;
		path = null;
		draw();
	}

	el.querySelector('.predict-btn').onclick = function () {
		predict(canvas);
	}

	const paths = [];

	let path;
	canvas.onmousedown = function (event) {
		if (event.button === 0 && !path) {
			path = [getPointer(event)];
			paths.push(path);
			draw();
		}
	}
	window.addEventListener('mousemove', event => {
		if (path) {
			path.push(getPointer(event));
			draw();
		}
	});
	window.addEventListener('mouseup', event => {
		if (event.button === 0) {
			path = null;
		}
	});

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

const canvas = document.getElementById('canvas');

const options = {
	antilias: true, 
	alpha: true
};
const gl = canvas.getContext('webgl', options) || canvas.getContext('experimental-webgl', options);

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
}
resize();

const Void = () => {}

const settingsEl = document.querySelector('.settings');

const settings = {
	trainingEnabled: true, 
	endlessTraining: false, 
	autoSaveCheckpoint: false, 
	learningRate: [0.01, 0.01, 1, 0.01], 
	checkpointSaveInterval: [0.1, 0.01, 1, 0.01], 
	batchSize: [1, 1, 666, 1], 
	trainSplit: [0.8, 0.01, 0.99, 0.01], 
	dataSplit: [1, 0.01, 1, 0.01]
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

function setSetting(key, value) {
	settings[key] = value;
	const el = document.getElementById(key);
	el.value = value;
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
	<div class="btn export-btn">export</div>
	<div class="btn import-btn">import</div>
</div>`);
settingsEl.appendChild(btnsEl);

btnsEl.querySelector('.export-btn').onclick = saveCheckpoint;

btnsEl.querySelector('.import-btn').onclick = function () {
	const el = document.createElement('input');
	el.type = 'file';
	el.accept = '.666';

	el.oninput = function (event) {
		const file = this.files[0];
		if (!file) return;

		const reader = new FileReader();
		reader.onload = function () {
			importCheckpoint(this.result);
		}
		reader.readAsText(file);
	}

	el.click();
}

// rendering

CanvasRenderingContext2D.prototype.scale2 = function (f) {
	this.scale(f, f);
}

const colors = {
	activation: '#ffeb3b', 
	label: '#fb382a'
};

let graphs;

function initGraphs() {
	graphs = {};

	const list = ['trainLoss', 'trainAccuracy', 'valLoss', 'valAccuracy', 'batchTime', 'epochTime'];

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

function drawHud(ctx) {
	const canvas = ctx.canvas;
	const scale = Math.max(canvas.width / 1366, canvas.height / 768);

	ctx.clearRect(0, 0, canvas.width, canvas.height);

	const W = canvas.width / scale;
	const H = canvas.height / scale;

	ctx.save();
	ctx.scale2(scale);

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

		ctx.fillStyle = '#888';
		ctx.font = 'normal 16px monospace';
		ctx.textBaseline = 'bottom';
		ctx.textAlign = 'right';
		const n = graph.points.length > 0 ? graph.points[graph.points.length - 1] : 0;
		ctx.fillText(n.toFixed(2), graphWidth, 0);

		ctx.fillStyle = '#fff';
		ctx.font = 'normal 10px monospace';
		ctx.textBaseline = 'top';
		ctx.textAlign = 'left';
		ctx.fillText(graph.name, 0, 7);

		ctx.restore();

		ctx.translate(graphWidth + 15, 0);
	}

	ctx.restore();

	//

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

	ctx.restore();
}

let now = 0;
let lastTime = Date.now();
let dt = 0;
let dts = 0;

let showT = 0;

function update() {
	now = Date.now();
	dt = now - lastTime;
	dts = dt / 1000;
	lastTime = now;

	const lf = getLerpFactor(0.1);
	showT = lerp(showT, 1, lf);
	headerEl.style.transform = `translateX(${(1 - showT) * 200}%)`;
	settingsEl.style.transform = `translateY(${(1 - showT) * -200}%)`;
	sketchEl.style.transform = `translateY(${(1 - showT) * 200}%)`;

	if (loaded && isTraining() && !epoching) {
		worker.postMessage({
			id: 'train'
		});
		epoching = true;
	}
}

function animate() {
	update();
	drawHud(hudCtx);
	window.requestAnimationFrame(animate);
}

animate();

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
