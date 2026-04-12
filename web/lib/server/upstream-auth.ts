import "server-only";

import type {
  ApiErrorBody,
  AdminQuoteListResponse,
  AdminUserListResponse,
  AuthCredentialsInput,
  AuthSessionResponse,
  EmailVerificationInput,
  SessionUser,
} from "@/lib/api-types";
import { UpstreamApiError, getServerAuthorizationHeader, getUpstreamApiBaseUrl } from "@/lib/server/upstream-quotes";

type RequestOptions = {
  method: "GET" | "POST" | "DELETE";
  sessionToken?: string | null;
  clientId?: string | null;
  body?: unknown;
};

async function upstreamRequest<T>(path: string, options: RequestOptions): Promise<T> {
  const authorizationHeaders = await getServerAuthorizationHeader();
  const headers: HeadersInit = {
    Accept: "application/json",
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.clientId ? { "X-Client-ID": options.clientId } : {}),
    ...(options.sessionToken ? { "X-Session-Token": options.sessionToken } : {}),
    ...authorizationHeaders,
  };

  const response = await fetch(`${getUpstreamApiBaseUrl()}${path}`, {
    method: options.method,
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
    cache: "no-store",
  });

  if (response.status === 204) {
    return null as T;
  }

  const text = await response.text();
  const parsedBody = text ? (JSON.parse(text) as ApiErrorBody | T) : null;

  if (!response.ok) {
    const detail =
      parsedBody && typeof parsedBody === "object" && "detail" in parsedBody
        ? String(parsedBody.detail)
        : response.statusText;
    throw new UpstreamApiError(detail || "La requete upstream a echoue.", response.status, parsedBody);
  }

  return parsedBody as T;
}

export function registerUpstreamAccount(body: AuthCredentialsInput, clientId?: string | null) {
  return upstreamRequest<AuthSessionResponse>("/auth/register", {
    method: "POST",
    body,
    clientId,
  });
}

export function loginUpstreamAccount(body: AuthCredentialsInput, clientId?: string | null) {
  return upstreamRequest<AuthSessionResponse>("/auth/login", {
    method: "POST",
    body,
    clientId,
  });
}

export function getUpstreamAuthSession(sessionToken?: string | null) {
  return upstreamRequest<AuthSessionResponse>("/auth/session", {
    method: "GET",
    sessionToken,
  });
}

export function logoutUpstreamAccount(sessionToken?: string | null) {
  return upstreamRequest<null>("/auth/logout", {
    method: "POST",
    sessionToken,
  });
}

export function verifyUpstreamAccountEmail(body: EmailVerificationInput) {
  return upstreamRequest<SessionUser>("/auth/verify-email", {
    method: "POST",
    body,
  });
}

export function listUpstreamAdminUsers(sessionToken: string) {
  return upstreamRequest<AdminUserListResponse>("/admin/users", {
    method: "GET",
    sessionToken,
  });
}

export function deactivateUpstreamAdminUser(userId: string, sessionToken: string) {
  return upstreamRequest<null>(`/admin/users/${userId}`, {
    method: "DELETE",
    sessionToken,
  });
}

export function listUpstreamAdminQuotes(sessionToken: string) {
  return upstreamRequest<AdminQuoteListResponse>("/admin/quotes", {
    method: "GET",
    sessionToken,
  });
}

export function deleteUpstreamAdminQuote(quoteId: string, sessionToken: string) {
  return upstreamRequest<null>(`/admin/quotes/${quoteId}`, {
    method: "DELETE",
    sessionToken,
  });
}
