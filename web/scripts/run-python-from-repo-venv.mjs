import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const webRoot = path.resolve(__dirname, "..");

const venvPython = process.platform === "win32"
  ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
  : path.join(repoRoot, ".venv", "bin", "python");

const pythonCommand = existsSync(venvPython) ? venvPython : "python";
const pythonArgs = process.argv.slice(2);

if (pythonArgs.length === 0) {
  console.error("Usage: node scripts/run-python-from-repo-venv.mjs <script> [...args]");
  process.exit(1);
}

const result = spawnSync(pythonCommand, pythonArgs, {
  cwd: webRoot,
  env: process.env,
  stdio: "inherit",
});

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}

process.exit(result.status ?? 1);
