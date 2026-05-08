import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import { defineConfig, loadEnv, Plugin } from 'vite';
import react from '@vitejs/plugin-react';

function resolveMcLnnPythonBin() {
  const candidates = [
    process.env.AQX_PYTHON_BIN,
    path.resolve(__dirname, 'python/mc_lnn_imputer/.venv/bin/python'),
    '/Users/HenokTeklu/miniconda3/envs/canalforge-clean/bin/python',
    'python3',
  ].filter((value): value is string => Boolean(value));

  for (const candidate of candidates) {
    if (candidate === 'python3' || fs.existsSync(candidate)) return candidate;
  }

  return 'python3';
}

function saveDataPlugin(): Plugin {
  return {
    name: 'save-data',
    configureServer(server) {
      // GET /api/regions — scan public/data/ subdirectories for region.json
      server.middlewares.use('/api/regions', (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        try {
          const dataDir = path.resolve(__dirname, 'public/data');
          const entries = fs.readdirSync(dataDir, { withFileTypes: true });
          const regions: any[] = [];
          for (const entry of entries) {
            if (!entry.isDirectory()) continue;
            const regionDir = path.join(dataDir, entry.name);
            const regionJsonPath = path.join(regionDir, 'region.json');
            if (fs.existsSync(regionJsonPath)) {
              try {
                const meta = JSON.parse(fs.readFileSync(regionJsonPath, 'utf-8'));
                // Scan for data_*.csv so the client can compute effective
                // data types without needing to probe each file individually.
                const dataFiles: string[] = [];
                try {
                  for (const f of fs.readdirSync(regionDir)) {
                    if (f.startsWith('data_') && f.endsWith('.csv')) dataFiles.push(f);
                  }
                } catch {}
                regions.push({ ...meta, dataFiles });
              } catch (e) {
                // skip malformed region.json
              }
            }
          }
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(regions));
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });

      // POST /api/delete-file — delete a single file within public/data/
      server.middlewares.use('/api/delete-file', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
        req.on('end', () => {
          try {
            const { filePath } = JSON.parse(body) as { filePath: string };
            const dataDir = path.resolve(__dirname, 'public/data');
            const fullPath = path.resolve(dataDir, filePath);
            // Safety: ensure we're deleting inside public/data
            if (!fullPath.startsWith(dataDir + path.sep) || fullPath === dataDir) {
              res.statusCode = 400;
              res.end(`Invalid path: ${filePath}`);
              return;
            }
            if (fs.existsSync(fullPath)) {
              fs.unlinkSync(fullPath);
              // Clean up empty parent directory (only within public/data/)
              const parentDir = path.dirname(fullPath);
              if (parentDir !== dataDir && parentDir.startsWith(dataDir + path.sep)) {
                try {
                  const remaining = fs.readdirSync(parentDir);
                  if (remaining.length === 0) fs.rmdirSync(parentDir);
                } catch { /* ignore */ }
              }
            }
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true }));
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });

      // Delete a region folder
      server.middlewares.use('/api/delete-folder', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
        req.on('end', () => {
          try {
            const { folder } = JSON.parse(body) as { folder: string };
            const dataDir = path.resolve(__dirname, 'public/data');
            const folderPath = path.resolve(dataDir, folder);
            // Safety: ensure we're deleting inside public/data
            if (!folderPath.startsWith(dataDir + path.sep) || folderPath === dataDir) {
              res.statusCode = 400;
              res.end(`Invalid folder: ${folder}`);
              return;
            }
            fs.rmSync(folderPath, { recursive: true, force: true });
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true }));
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });

      // GET /api/list-rasters?region={id} — list raster analysis metadata
      server.middlewares.use('/api/list-rasters', (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        try {
          const url = new URL(req.url || '', 'http://localhost');
          const regionId = url.searchParams.get('region');
          if (!regionId) {
            res.statusCode = 400;
            res.end('Missing region parameter');
            return;
          }
          const dataDir = path.resolve(__dirname, 'public/data');
          const regionDir = path.resolve(dataDir, regionId);
          if (!regionDir.startsWith(dataDir + path.sep)) {
            res.statusCode = 400;
            res.end('Invalid region');
            return;
          }
          const results: any[] = [];
          if (fs.existsSync(regionDir)) {
            // Scan subdirectories for raster_*.json (new layout)
            for (const sub of fs.readdirSync(regionDir, { withFileTypes: true })) {
              if (!sub.isDirectory()) continue;
              const subDir = path.join(regionDir, sub.name);
              for (const file of fs.readdirSync(subDir)) {
                if (file.startsWith('raster_') && file.endsWith('.json')) {
                  try {
                    const data = JSON.parse(fs.readFileSync(path.join(subDir, file), 'utf-8'));
                    // Parse dataType from filename: raster_{dataType}_{code}.json
                    const match = file.match(/^raster_([a-z0-9_]+?)_(.+)\.json$/);
                    const filePath = `${regionId}/${sub.name}/${file}`;
                    results.push({
                      title: data.title || file,
                      code: data.code || (match ? match[2] : file.replace('.json', '')),
                      aquiferId: data.aquiferId || '',
                      aquiferName: data.aquiferName || '',
                      regionId: data.regionId || regionId,
                      filePath,
                      dataType: data.dataType || (match ? match[1] : 'wte'),
                      params: data.params || {},
                      createdAt: data.createdAt || '',
                      options: data.options || undefined,
                      generatedAt: data.generatedAt || undefined,
                    });
                  } catch { /* skip malformed */ }
                }
              }
            }
          }
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(results));
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });

      // POST /api/rename-raster — rename a raster analysis file
      server.middlewares.use('/api/rename-raster', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
        req.on('end', () => {
          try {
            const { oldPath, newPath, newCode, newTitle } = JSON.parse(body) as {
              oldPath: string; newPath: string; newCode: string; newTitle: string;
            };
            const dataDir = path.resolve(__dirname, 'public/data');
            const fullOldPath = path.resolve(dataDir, oldPath);
            const fullNewPath = path.resolve(dataDir, newPath);

            // Safety: ensure both paths are within public/data
            if (!fullOldPath.startsWith(dataDir + path.sep) || !fullNewPath.startsWith(dataDir + path.sep)) {
              res.statusCode = 400;
              res.end('Invalid path');
              return;
            }

            if (!fs.existsSync(fullOldPath)) {
              res.statusCode = 404;
              res.end('File not found');
              return;
            }

            // Read, update, write
            const data = JSON.parse(fs.readFileSync(fullOldPath, 'utf-8'));
            data.code = newCode;
            data.title = newTitle;

            fs.mkdirSync(path.dirname(fullNewPath), { recursive: true });
            fs.writeFileSync(fullNewPath, JSON.stringify(data), 'utf-8');

            // Delete old file if path changed
            if (fullOldPath !== fullNewPath) {
              fs.unlinkSync(fullOldPath);
              // Clean up empty parent directory
              const parentDir = path.dirname(fullOldPath);
              if (parentDir !== dataDir && parentDir.startsWith(dataDir + path.sep)) {
                try {
                  const remaining = fs.readdirSync(parentDir);
                  if (remaining.length === 0) fs.rmdirSync(parentDir);
                } catch { /* ignore */ }
              }
            }

            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true }));
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });

      // GET /api/list-models?region={id} — list imputation model metadata
      server.middlewares.use('/api/list-models', (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        try {
          const url = new URL(req.url || '', 'http://localhost');
          const regionId = url.searchParams.get('region');
          if (!regionId) {
            res.statusCode = 400;
            res.end('Missing region parameter');
            return;
          }
          const dataDir = path.resolve(__dirname, 'public/data');
          const regionDir = path.resolve(dataDir, regionId);
          if (!regionDir.startsWith(dataDir + path.sep)) {
            res.statusCode = 400;
            res.end('Invalid region');
            return;
          }
          const results: any[] = [];
          if (fs.existsSync(regionDir)) {
            for (const sub of fs.readdirSync(regionDir, { withFileTypes: true })) {
              if (!sub.isDirectory()) continue;
              const subDir = path.join(regionDir, sub.name);
              for (const file of fs.readdirSync(subDir)) {
                if (file.startsWith('model_') && file.endsWith('.json')) {
                  try {
                    const data = JSON.parse(fs.readFileSync(path.join(subDir, file), 'utf-8'));
                    const filePath = `${regionId}/${sub.name}/${file}`;
                    // Strip data and log arrays for performance
                    results.push({
                      title: data.title || file,
                      code: data.code || file.replace('.json', ''),
                      method: data.method || 'original-elm',
                      aquiferId: data.aquiferId || '',
                      aquiferName: data.aquiferName || '',
                      regionId: data.regionId || regionId,
                      filePath,
                      dataType: data.dataType || 'wte',
                      params: data.params || {},
                      createdAt: data.createdAt || '',
                      wellMetrics: data.wellMetrics || {},
                    });
                  } catch { /* skip malformed */ }
                }
              }
            }
          }
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(results));
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });

      // GET /api/gldas-proxy?url={encodedUrl} — proxy GLDAS THREDDS requests for CORS
      // Proxy for the Water Quality Portal. WQP doesn't send
      // Access-Control-Expose-Headers, so the browser hides count
      // headers (Total-Result-Count, Total-Site-Count, etc.) from JS
      // even though they're in the response. We forward the request
      // server-side, then re-emit those headers with the proper expose
      // header. Also avoids future CORS surprises like the HEAD-403
      // gotcha.
      server.middlewares.use('/api/wqp-proxy', async (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        try {
          const url = new URL(req.url || '', 'http://localhost');
          const targetUrl = url.searchParams.get('url');
          const headersOnly = url.searchParams.get('headersOnly') === '1';
          if (!targetUrl || !targetUrl.startsWith('https://www.waterqualitydata.us/')) {
            res.statusCode = 400;
            res.end('Missing or non-WQP url parameter');
            return;
          }
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 120000);
          try {
            const response = await fetch(targetUrl, { signal: controller.signal });
            clearTimeout(timeout);
            // Copy headers we care about (counts + content)
            const passthrough = [
              'content-type',
              'total-site-count', 'nwis-site-count', 'storet-site-count',
              'total-activity-count', 'nwis-activity-count', 'storet-activity-count',
              'total-result-count', 'nwis-result-count', 'storet-result-count',
            ];
            for (const h of passthrough) {
              const v = response.headers.get(h);
              if (v) res.setHeader(h, v);
            }
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Expose-Headers', passthrough.join(','));
            res.statusCode = response.status;
            if (headersOnly) {
              // Cancel the upstream body stream so we don't pay for the
              // full CSV download just to read count headers.
              try { await response.body?.cancel(); } catch {}
              res.end('');
            } else {
              const body = await response.text();
              res.end(body);
            }
          } catch (fetchErr: any) {
            clearTimeout(timeout);
            res.statusCode = 502;
            res.end(`WQP proxy error: ${fetchErr.message || fetchErr}`);
          }
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });

      server.middlewares.use('/api/gldas-proxy', async (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        try {
          const url = new URL(req.url || '', 'http://localhost');
          const targetUrl = url.searchParams.get('url');
          if (!targetUrl) {
            res.statusCode = 400;
            res.end('Missing url parameter');
            return;
          }
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 15000);
          try {
            const response = await fetch(targetUrl, { signal: controller.signal });
            clearTimeout(timeout);
            const contentType = response.headers.get('content-type') || 'text/plain';
            res.setHeader('Content-Type', contentType);
            res.setHeader('Access-Control-Allow-Origin', '*');
            const body = await response.text();
            res.end(body);
          } catch (fetchErr: any) {
            clearTimeout(timeout);
            res.statusCode = 502;
            res.end(`GLDAS proxy error: ${fetchErr.message || fetchErr}`);
          }
        } catch (err) {
          res.statusCode = 500;
          res.end(String(err));
        }
      });

      // POST /api/rename-model — rename an imputation model file
      server.middlewares.use('/api/rename-model', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
        req.on('end', () => {
          try {
            const { oldPath, newPath, newCode, newTitle } = JSON.parse(body) as {
              oldPath: string; newPath: string; newCode: string; newTitle: string;
            };
            const dataDir = path.resolve(__dirname, 'public/data');
            const fullOldPath = path.resolve(dataDir, oldPath);
            const fullNewPath = path.resolve(dataDir, newPath);

            if (!fullOldPath.startsWith(dataDir + path.sep) || !fullNewPath.startsWith(dataDir + path.sep)) {
              res.statusCode = 400;
              res.end('Invalid path');
              return;
            }

            if (!fs.existsSync(fullOldPath)) {
              res.statusCode = 404;
              res.end('File not found');
              return;
            }

            const data = JSON.parse(fs.readFileSync(fullOldPath, 'utf-8'));
            data.code = newCode;
            data.title = newTitle;

            fs.mkdirSync(path.dirname(fullNewPath), { recursive: true });
            fs.writeFileSync(fullNewPath, JSON.stringify(data), 'utf-8');

            if (fullOldPath !== fullNewPath) {
              fs.unlinkSync(fullOldPath);
              const parentDir = path.dirname(fullOldPath);
              if (parentDir !== dataDir && parentDir.startsWith(dataDir + path.sep)) {
                try {
                  const remaining = fs.readdirSync(parentDir);
                  if (remaining.length === 0) fs.rmdirSync(parentDir);
                } catch { /* ignore */ }
              }
            }

            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true }));
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });

      server.middlewares.use('/api/save-data', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
        req.on('end', () => {
          try {
            const { files } = JSON.parse(body) as { files: { path: string; content: string }[] };
            const dataDir = path.resolve(__dirname, 'public/data');
            for (const file of files) {
              const filePath = path.resolve(dataDir, file.path);
              // Safety: ensure we're writing inside public/data
              if (!filePath.startsWith(dataDir)) {
                res.statusCode = 400;
                res.end(`Invalid path: ${file.path}`);
                return;
              }
              fs.mkdirSync(path.dirname(filePath), { recursive: true });
              fs.writeFileSync(filePath, file.content, 'utf-8');
            }
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true, count: files.length }));
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });

      server.middlewares.use('/api/impute-mc-lnn', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
        req.on('end', () => {
          try {
            const payload = JSON.parse(body);
            const scriptPath = path.resolve(__dirname, 'python/mc_lnn_imputer/app/browser_mc_lnn_request.py');
            const pythonBin = resolveMcLnnPythonBin();
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            const child = spawn(pythonBin, [scriptPath], {
              cwd: __dirname,
              env: { ...process.env, PYTHONUNBUFFERED: '1' },
            });
            let stderr = '';
            child.stdout.on('data', chunk => { res.write(chunk); });
            child.stderr.on('data', chunk => { stderr += chunk.toString(); });
            child.on('error', err => {
              res.write(JSON.stringify({ type: 'error', message: `Failed to start Python MC+LNN backend: ${String(err)}` }) + '\n');
              res.end();
            });
            child.on('close', code => {
              if (code !== 0) {
                res.write(JSON.stringify({ type: 'error', message: stderr || `Python MC+LNN backend exited with code ${code}` }) + '\n');
              }
              res.end();
            });
            child.stdin.write(JSON.stringify(payload));
            child.stdin.end();
          } catch (err) {
            res.statusCode = 500;
            res.end(String(err));
          }
        });
      });
    }
  };
}

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react(), saveDataPlugin()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.USGS_API_KEY': JSON.stringify(env.USGS_API_KEY || '')
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
