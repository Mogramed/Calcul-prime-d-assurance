import "server-only";

import { cookies } from "next/headers";
import type { NextResponse } from "next/server";

const AUTH_SESSION_COOKIE_NAME = "nova_session";
const AUTH_SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30;

function isSecureRequest(request: Request) {
  if (process.env.COOKIE_SECURE === "true") {
    return true;
  }

  if (process.env.COOKIE_SECURE === "false") {
    return false;
  }

  const forwardedProto = request.headers.get("x-forwarded-proto");
  if (forwardedProto) {
    return forwardedProto.includes("https");
  }

  return new URL(request.url).protocol === "https:";
}

export type AuthCookieSession = {
  sessionToken: string | null;
  attach(response: NextResponse, sessionToken: string): NextResponse;
  clear(response: NextResponse): NextResponse;
};

export async function getAuthCookieSession(request: Request): Promise<AuthCookieSession> {
  const cookieStore = await cookies();
  const existing = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value ?? null;
  const secure = isSecureRequest(request);

  return {
    sessionToken: existing,
    attach(response, sessionToken) {
      response.cookies.set({
        name: AUTH_SESSION_COOKIE_NAME,
        value: sessionToken,
        httpOnly: true,
        sameSite: "lax",
        secure,
        maxAge: AUTH_SESSION_COOKIE_MAX_AGE_SECONDS,
        path: "/",
      });
      return response;
    },
    clear(response) {
      response.cookies.set({
        name: AUTH_SESSION_COOKIE_NAME,
        value: "",
        httpOnly: true,
        sameSite: "lax",
        secure,
        maxAge: 0,
        path: "/",
      });
      return response;
    },
  };
}

export async function readAuthSessionToken() {
  const cookieStore = await cookies();
  return cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value ?? null;
}
