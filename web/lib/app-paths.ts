const rawBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const appBasePath = normalizeBasePath(rawBasePath);
export const bffBasePath = `${appBasePath}/app-api`;

export function withAppBasePath(path: string) {
  const normalizedPath = normalizePath(path);
  return `${appBasePath}${normalizedPath}`;
}

export function withBffPath(path: string) {
  const normalizedPath = normalizePath(path);
  return `${bffBasePath}${normalizedPath}`;
}

function normalizeBasePath(value: string) {
  const trimmed = value.trim();
  if (!trimmed || trimmed === "/") {
    return "";
  }

  const prefixed = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return prefixed.endsWith("/") ? prefixed.slice(0, -1) : prefixed;
}

function normalizePath(path: string) {
  if (!path) {
    return "";
  }

  return path.startsWith("/") ? path : `/${path}`;
}
