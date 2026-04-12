import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse, publicSessionResponse } from "@/lib/server/bff-response";
import { getUpstreamAuthSession } from "@/lib/server/upstream-auth";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const authSession = await getAuthCookieSession(request);

  try {
    const session = await getUpstreamAuthSession(authSession.sessionToken);
    const response = NextResponse.json(publicSessionResponse(session));
    if (!session.authenticated && authSession.sessionToken) {
      authSession.clear(response);
    }
    return response;
  } catch (error) {
    const response = errorJsonResponse(error, "La session ne peut pas etre verifiee pour le moment.");
    authSession.clear(response);
    return response;
  }
}
