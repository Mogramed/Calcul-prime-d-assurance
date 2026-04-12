import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { getClientSession } from "@/lib/server/client-session";
import { getUpstreamQuote, updateUpstreamQuote } from "@/lib/server/upstream-quotes";
import { NextResponse } from "next/server";
import type { PredictionInput } from "@/generated/client/types.gen";

export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ quoteId: string }> },
) {
  const authSession = await getAuthCookieSession(request);
  if (!authSession.sessionToken) {
    return NextResponse.json(
      {
        detail: "Veuillez vous connecter pour acceder a vos devis.",
      },
      { status: 401 },
    );
  }
  const session = await getClientSession(request);

  try {
    const { quoteId } = await context.params;
    const quote = await getUpstreamQuote(quoteId, session.clientId, authSession.sessionToken);
    return session.attach(NextResponse.json(quote));
  } catch (error) {
    return session.attach(errorJsonResponse(error, "Le service de devis est temporairement indisponible."));
  }
}

export async function PUT(
  request: Request,
  context: { params: Promise<{ quoteId: string }> },
) {
  const authSession = await getAuthCookieSession(request);
  if (!authSession.sessionToken) {
    return NextResponse.json(
      {
        detail: "Veuillez vous connecter pour modifier ce devis.",
      },
      { status: 401 },
    );
  }
  const session = await getClientSession(request);

  try {
    const { quoteId } = await context.params;
    const payload = (await request.json()) as PredictionInput;
    const quote = await updateUpstreamQuote(quoteId, payload, session.clientId, authSession.sessionToken);
    return session.attach(NextResponse.json(quote));
  } catch (error) {
    return session.attach(errorJsonResponse(error, "Le service de devis est temporairement indisponible."));
  }
}
