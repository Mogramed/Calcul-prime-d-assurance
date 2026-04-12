import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { listUpstreamAdminQuotes } from "@/lib/server/upstream-auth";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const authSession = await getAuthCookieSession(request);

  try {
    if (!authSession.sessionToken) {
      return NextResponse.json({ detail: "Authentication is required." }, { status: 401 });
    }
    const quotes = await listUpstreamAdminQuotes(authSession.sessionToken);
    return NextResponse.json(quotes);
  } catch (error) {
    return errorJsonResponse(error, "La console admin est temporairement indisponible.");
  }
}
