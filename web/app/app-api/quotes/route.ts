import type { PredictionInput } from "@/generated/client/types.gen";
import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { getClientSession } from "@/lib/server/client-session";
import { NextResponse } from "next/server";
import { createUpstreamQuote, listUpstreamQuotes } from "@/lib/server/upstream-quotes";

export const runtime = "nodejs";

export async function GET(request: Request) {
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
    const quotes = await listUpstreamQuotes(session.clientId, authSession.sessionToken);
    return session.attach(NextResponse.json(quotes));
  } catch (error) {
    return session.attach(errorJsonResponse(error, "Le service de devis est temporairement indisponible."));
  }
}

export async function POST(request: Request) {
  const authSession = await getAuthCookieSession(request);
  if (!authSession.sessionToken) {
    return NextResponse.json(
      {
        detail: "Veuillez vous connecter pour creer un devis.",
      },
      { status: 401 },
    );
  }
  const session = await getClientSession(request);

  try {
    const payload = (await request.json()) as PredictionInput;
    const quote = await createUpstreamQuote(payload, session.clientId, authSession.sessionToken);
    return session.attach(NextResponse.json(quote));
  } catch (error) {
    return session.attach(errorJsonResponse(error, "Le service de devis est temporairement indisponible."));
  }
}
