import "server-only";

import type {
  ApiErrorResponse,
  PredictionInput,
  QuoteListResponse,
  QuoteResponse,
} from "@/generated/client/types.gen";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const METADATA_IDENTITY_TOKEN_URL =
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity";

export class UpstreamApiError extends Error {
  status: number;
  body: ApiErrorResponse | unknown;

  constructor(message: string, status: number, body: ApiErrorResponse | unknown) {
    super(message);
    this.name = "UpstreamApiError";
    this.status = status;
    this.body = body;
  }
}

type RequestOptions = {
  method: "GET" | "POST";
  clientId: string;
  sessionToken?: string | null;
  body?: PredictionInput;
};

export function getUpstreamApiBaseUrl() {
  return (process.env.API_BASE_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, "");
}

export async function getServerAuthorizationHeader() {
  const audience = process.env.API_AUDIENCE;
  if (!audience) {
    return {} as Record<string, string>;
  }

  const response = await fetch(
    `${METADATA_IDENTITY_TOKEN_URL}?audience=${encodeURIComponent(audience)}`,
    {
      headers: {
        "Metadata-Flavor": "Google",
      },
      cache: "no-store",
    },
  );

  if (!response.ok) {
    throw new Error("Impossible d'obtenir un jeton d'acces serveur pour joindre l'API privee.");
  }

  const token = await response.text();
  return {
    Authorization: `Bearer ${token}`,
  } satisfies Record<string, string>;
}

async function upstreamRequest<T>(path: string, options: RequestOptions): Promise<T> {
  const authorizationHeaders = await getServerAuthorizationHeader();
  const headers: HeadersInit = {
    Accept: "application/json",
    "X-Client-ID": options.clientId,
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.sessionToken ? { "X-Session-Token": options.sessionToken } : {}),
    ...authorizationHeaders,
  };

  const response = await fetch(`${getUpstreamApiBaseUrl()}${path}`, {
    method: options.method,
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
    cache: "no-store",
  });

  const text = await response.text();
  const parsedBody = text ? (JSON.parse(text) as ApiErrorResponse | T) : null;

  if (!response.ok) {
    const detail =
      parsedBody && typeof parsedBody === "object" && "detail" in parsedBody
        ? String(parsedBody.detail)
        : response.statusText;

    throw new UpstreamApiError(detail || "La requete upstream a echoue.", response.status, parsedBody);
  }

  return parsedBody as T;
}

async function upstreamBinaryRequest(path: string, options: Omit<RequestOptions, "body">) {
  const authorizationHeaders = await getServerAuthorizationHeader();
  const headers: HeadersInit = {
    Accept: "application/pdf",
    "X-Client-ID": options.clientId,
    ...(options.sessionToken ? { "X-Session-Token": options.sessionToken } : {}),
    ...authorizationHeaders,
  };

  const response = await fetch(`${getUpstreamApiBaseUrl()}${path}`, {
    method: options.method,
    headers,
    cache: "no-store",
  });

  if (!response.ok) {
    const text = await response.text();
    const parsedBody = text ? (JSON.parse(text) as ApiErrorResponse) : null;
    const detail =
      parsedBody && typeof parsedBody === "object" && "detail" in parsedBody
        ? String(parsedBody.detail)
        : response.statusText;
    throw new UpstreamApiError(detail || "La requete upstream a echoue.", response.status, parsedBody);
  }

  return response;
}

export function createUpstreamQuote(
  body: PredictionInput,
  clientId: string,
  sessionToken?: string | null,
) {
  return upstreamRequest<QuoteResponse>("/quotes", {
    method: "POST",
    body,
    clientId,
    sessionToken,
  });
}

export function listUpstreamQuotes(clientId: string, sessionToken?: string | null) {
  return upstreamRequest<QuoteListResponse>("/quotes", {
    method: "GET",
    clientId,
    sessionToken,
  });
}

export function getUpstreamQuote(quoteId: string, clientId: string, sessionToken?: string | null) {
  return upstreamRequest<QuoteResponse>(`/quotes/${quoteId}`, {
    method: "GET",
    clientId,
    sessionToken,
  });
}

export function downloadUpstreamQuoteReport(
  quoteId: string,
  clientId: string,
  sessionToken?: string | null,
) {
  return upstreamBinaryRequest(`/quotes/${quoteId}/report.pdf`, {
    method: "GET",
    clientId,
    sessionToken,
  });
}
