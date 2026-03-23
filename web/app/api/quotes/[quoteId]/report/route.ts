import { getAuthCookieSession } from "@/lib/server/auth-session";
import { errorJsonResponse } from "@/lib/server/bff-response";
import { getClientSession } from "@/lib/server/client-session";
import { downloadUpstreamQuoteReport } from "@/lib/server/upstream-quotes";
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
    const upstreamResponse = await downloadUpstreamQuoteReport(
      quoteId,
      session.clientId,
      authSession.sessionToken,
    );
    const pdfBytes = await upstreamResponse.arrayBuffer();

    const response = new NextResponse(pdfBytes, {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition":
          upstreamResponse.headers.get("content-disposition") ??
          `attachment; filename="nova-devis-${quoteId}.pdf"`,
      },
    });
    return session.attach(response);
  } catch (error) {
    return session.attach(errorJsonResponse(error, "Le rapport PDF est temporairement indisponible."));
  }
}
