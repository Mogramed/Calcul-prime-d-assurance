import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse, publicSessionResponse } from "@/lib/server/bff-response";
import { getClientSession } from "@/lib/server/client-session";
import { registerUpstreamAccount } from "@/lib/server/upstream-auth";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const clientSession = await getClientSession(request);
  const authSession = await getAuthCookieSession(request);
  const requestUrl = new URL(request.url);
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  const publicWebUrl = `${requestUrl.origin}${basePath}`.replace(/\/$/, "");

  try {
    const payload = (await request.json()) as { email: string; password: string };
    const session = await registerUpstreamAccount(payload, clientSession.clientId, publicWebUrl);
    const response = clientSession.attach(NextResponse.json(publicSessionResponse(session)));
    if (session.session_token) {
      authSession.attach(response, session.session_token);
    } else {
      authSession.clear(response);
    }
    return response;
  } catch (error) {
    return clientSession.attach(errorJsonResponse(error, "L'inscription est temporairement indisponible."));
  }
}
