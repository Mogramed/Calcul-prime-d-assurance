import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { deleteUpstreamAdminQuote } from "@/lib/server/upstream-auth";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function DELETE(
  request: Request,
  context: { params: Promise<{ quoteId: string }> },
) {
  const authSession = await getAuthCookieSession(request);

  try {
    if (!authSession.sessionToken) {
      return NextResponse.json({ detail: "Authentication is required." }, { status: 401 });
    }
    const { quoteId } = await context.params;
    await deleteUpstreamAdminQuote(quoteId, authSession.sessionToken);
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    return errorJsonResponse(error, "La console admin est temporairement indisponible.");
  }
}
