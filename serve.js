const http = require('http');
const fs = require('fs');

const mimeTypes = {
	'html': 'text/html', 
	'js': 'text/javascript', 
	'css': 'text/css', 
	'csv': 'text/plain'
};

const server = http.createServer(function (req, res) {
	if (req.method === 'GET') {
		let path = new URL('http://666.XD' + req.url).pathname;
		if (path === '/') path += 'index.html';
		path = '.' + path;

		fs.exists(path, yes => {
			if (yes) {
				res.writeHead(200, {
					'Content-Type': mimeTypes[path.slice(path.lastIndexOf('.') + 1)] || 'application/octet-stream', 
					'Cross-Origin-Opener-Policy': 'same-origin', 
					'Cross-Origin-Embedder-Policy': 'require-corp', 
					'Access-Control-Allow-Origin': '*', 
					'Access-Control-Allow-Headers': '*'
				});
				fs.createReadStream(path).pipe(res);
			} else {
				res.writeHead(404).end();
			}
		})
		return;
	}

	return res.writeHead(404).end();
});

const port = 4666;
server.listen(port, () => console.log(`server running on port http://localhost:${port}...`));
