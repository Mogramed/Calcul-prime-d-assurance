import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { listUpstreamAdminUsers } from "@/lib/server/upstream-auth";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const authSession = await getAuthCookieSession(request);

  try {
    if (!authSession.sessionToken) {
      return NextResponse.json({ detail: "Authentication is required." }, { status: 401 });
    }
    const users = await listUpstreamAdminUsers(authSession.sessionToken);
    return NextResponse.json(users);
  } catch (error) {
    return errorJsonResponse(error, "La console admin est temporairement indisponible.");
  }
}
