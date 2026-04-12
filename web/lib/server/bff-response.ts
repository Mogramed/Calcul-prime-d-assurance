import { NextResponse } from "next/server";

import type { AuthSessionResponse } from "@/lib/api-types";
import { UpstreamApiError } from "@/lib/server/upstream-quotes";

export function errorJsonResponse(error: unknown, fallbackDetail: string) {
  if (error instanceof UpstreamApiError) {
    return NextResponse.json(
      {
        detail: error.message,
      },
      { status: error.status },
    );
  }

  return NextResponse.json(
    {
      detail: fallbackDetail,
    },
    { status: 500 },
  );
}

export function publicSessionResponse(session: AuthSessionResponse): AuthSessionResponse {
  return {
    authenticated: session.authenticated,
    user: session.user,
    expires_at_utc: session.expires_at_utc ?? null,
    email_verification_required: session.email_verification_required ?? false,
    email_verification_delivery: session.email_verification_delivery ?? null,
  };
}
