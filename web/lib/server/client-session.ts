import "server-only";

import { randomUUID } from "node:crypto";

import { cookies } from "next/headers";
import type { NextResponse } from "next/server";

const CLIENT_ID_COOKIE_NAME = "nova_client_id";
const CLIENT_ID_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;

export type ClientSession = {
  clientId: string;
  attach(response: NextResponse): NextResponse;
};

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

export async function getClientSession(request: Request): Promise<ClientSession> {
  const cookieStore = await cookies();
  const existing = cookieStore.get(CLIENT_ID_COOKIE_NAME)?.value;
  const clientId = existing ?? randomUUID();
  const secure = isSecureRequest(request);

  return {
    clientId,
    attach(response) {
      if (!existing) {
        response.cookies.set({
          name: CLIENT_ID_COOKIE_NAME,
          value: clientId,
          httpOnly: true,
          sameSite: "lax",
          secure,
          maxAge: CLIENT_ID_COOKIE_MAX_AGE_SECONDS,
          path: "/",
        });
      }

      return response;
    },
  };
}
