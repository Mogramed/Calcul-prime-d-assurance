import type { ApiErrorResponse } from "./types.gen";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export const API_BASE_URL = (
  process.env.API_BASE_URL ?? DEFAULT_API_BASE_URL
).replace(/\/$/, "");

export class ApiClientError extends Error {
  status: number;
  body: ApiErrorResponse | unknown;

  constructor(message: string, status: number, body: ApiErrorResponse | unknown) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.body = body;
  }
}

type RequestOptions = RequestInit & {
  json?: unknown;
};

export async function request<T>(path: string, options: RequestOptions = {}) {
  const { json, headers, ...rest } = options;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    headers: {
      Accept: "application/json",
      ...(json ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: json ? JSON.stringify(json) : rest.body,
  });

  const text = await response.text();
  const body = text ? (JSON.parse(text) as ApiErrorResponse | T) : null;

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String(body.detail)
        : response.statusText;
    throw new ApiClientError(detail || "La requete a echoue.", response.status, body);
  }

  return body as T;
}
