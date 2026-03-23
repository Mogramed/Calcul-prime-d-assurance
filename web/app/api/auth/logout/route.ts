import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { logoutUpstreamAccount } from "@/lib/server/upstream-auth";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const authSession = await getAuthCookieSession(request);

  try {
    await logoutUpstreamAccount(authSession.sessionToken);
    return authSession.clear(new NextResponse(null, { status: 204 }));
  } catch (error) {
    return authSession.clear(errorJsonResponse(error, "La deconnexion est temporairement indisponible."));
  }
}
