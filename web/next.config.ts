import path from "node:path";
import { fileURLToPath } from "node:url";

import type { NextConfig } from "next";

function normalizeBasePath(value: string | undefined) {
  const trimmed = value?.trim() ?? "";
  if (!trimmed || trimmed === "/") {
    return "";
  }

  const prefixed = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return prefixed.endsWith("/") ? prefixed.slice(0, -1) : prefixed;
}

const basePath = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);
const workspaceRoot = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  output: "standalone",
  turbopack: {
    root: workspaceRoot,
  },
  ...(basePath ? { basePath } : {}),
};

export default nextConfig;
