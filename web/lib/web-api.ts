import type {
  PredictionInput,
  QuoteListResponse,
  QuoteResponse,
} from "@/generated/client/types.gen";
import type {
  ApiErrorBody,
  AdminQuoteListResponse,
  AdminUserListResponse,
  AuthCredentialsInput,
  AuthSessionResponse,
} from "@/lib/api-types";

export class WebApiError extends Error {
  status: number;
  body: ApiErrorBody | unknown;

  constructor(message: string, status: number, body: ApiErrorBody | unknown) {
    super(message);
    this.name = "WebApiError";
    this.status = status;
    this.body = body;
  }
}

type RequestOptions = RequestInit & {
  json?: unknown;
};

function friendlyErrorMessage(status: number, body: ApiErrorBody | unknown) {
  if (status === 400 || status === 422) {
    return "Certaines informations doivent etre verifiees avant de continuer.";
  }

  if (status === 401) {
    return "Veuillez vous connecter pour continuer.";
  }

  if (status === 403) {
    return "Vous n'avez pas les autorisations necessaires pour cette action.";
  }

  if (status === 404) {
    return "Cet element n'est plus disponible.";
  }

  if (status === 409) {
    return "Un compte existe deja avec cette adresse email.";
  }

  if (status === 503) {
    return "Le service est temporairement indisponible.";
  }

  if (body && typeof body === "object" && "detail" in body && typeof body.detail === "string") {
    return body.detail;
  }

  return "Une erreur est survenue. Merci de reessayer dans un instant.";
}

async function request<T>(path: string, options: RequestOptions = {}) {
  const { json, headers, ...rest } = options;
  const response = await fetch(path, {
    ...rest,
    headers: {
      Accept: "application/json",
      ...(json ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: json ? JSON.stringify(json) : rest.body,
  });

  if (response.status === 204) {
    return null as T;
  }

  const text = await response.text();
  const body = text ? (JSON.parse(text) as ApiErrorBody | T) : null;

  if (!response.ok) {
    throw new WebApiError(friendlyErrorMessage(response.status, body), response.status, body);
  }

  return body as T;
}

export function createQuote(body: PredictionInput) {
  return request<QuoteResponse>("/api/quotes", {
    method: "POST",
    json: body,
  });
}

export function listQuotes(signal?: AbortSignal) {
  return request<QuoteListResponse>("/api/quotes", {
    method: "GET",
    signal,
  });
}

export function getQuote(quoteId: string, signal?: AbortSignal) {
  return request<QuoteResponse>(`/api/quotes/${quoteId}`, {
    method: "GET",
    signal,
  });
}

export function registerAccount(body: AuthCredentialsInput) {
  return request<AuthSessionResponse>("/api/auth/register", {
    method: "POST",
    json: body,
  });
}

export function loginAccount(body: AuthCredentialsInput) {
  return request<AuthSessionResponse>("/api/auth/login", {
    method: "POST",
    json: body,
  });
}

export function getAuthSession(signal?: AbortSignal) {
  return request<AuthSessionResponse>("/api/auth/session", {
    method: "GET",
    signal,
  });
}

export function logoutAccount() {
  return request<null>("/api/auth/logout", {
    method: "POST",
  });
}

export function listAdminUsers(signal?: AbortSignal) {
  return request<AdminUserListResponse>("/api/admin/users", {
    method: "GET",
    signal,
  });
}

export function deactivateAdminUser(userId: string) {
  return request<null>(`/api/admin/users/${userId}`, {
    method: "DELETE",
  });
}

export function listAdminQuotes(signal?: AbortSignal) {
  return request<AdminQuoteListResponse>("/api/admin/quotes", {
    method: "GET",
    signal,
  });
}

export function deleteAdminQuote(quoteId: string) {
  return request<null>(`/api/admin/quotes/${quoteId}`, {
    method: "DELETE",
  });
}

export function getQuoteReportUrl(quoteId: string) {
  return `/api/quotes/${quoteId}/report`;
}
