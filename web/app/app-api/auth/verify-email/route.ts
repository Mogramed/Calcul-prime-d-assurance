import { errorJsonResponse } from "@/lib/server/bff-response";
import { getClientSession } from "@/lib/server/client-session";
import { verifyUpstreamAccountEmail } from "@/lib/server/upstream-auth";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const clientSession = await getClientSession(request);

  try {
    const payload = (await request.json()) as { token: string };
    const user = await verifyUpstreamAccountEmail(payload);
    return clientSession.attach(NextResponse.json(user));
  } catch (error) {
    return clientSession.attach(
      errorJsonResponse(error, "La verification de l'adresse email est temporairement indisponible."),
    );
  }
}
