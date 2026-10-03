// Static-site build for Vercel.
// Reads VITE_API_URL from the environment and bakes it into
// dist/config.js, then copies the static assets to dist/.
const fs = require("fs");
const path = require("path");

const apiUrl = process.env.VITE_API_URL || "http://localhost:8000";
const dist = path.join(__dirname, "dist");

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });

fs.writeFileSync(
  path.join(dist, "config.js"),
  `window.FRUIT_API_URL = ${JSON.stringify(apiUrl)};\n`
);

for (const file of ["index.html", "styles.css", "app.js"]) {
  fs.copyFileSync(path.join(__dirname, file), path.join(dist, file));
}

console.log(`Built frontend to dist/ with API URL: ${apiUrl}`);
