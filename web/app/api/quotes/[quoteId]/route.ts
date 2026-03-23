import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { getClientSession } from "@/lib/server/client-session";
import { getUpstreamQuote } from "@/lib/server/upstream-quotes";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ quoteId: string }> },
) {
  const session = await getClientSession(request);
  const authSession = await getAuthCookieSession(request);

  try {
    const { quoteId } = await context.params;
    const quote = await getUpstreamQuote(quoteId, session.clientId, authSession.sessionToken);
    return session.attach(NextResponse.json(quote));
  } catch (error) {
    return session.attach(errorJsonResponse(error, "Le service de devis est temporairement indisponible."));
  }
}
