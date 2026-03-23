import "server-only";

import { readAuthSessionToken } from "@/lib/server/auth-session";
import { getUpstreamAuthSession } from "@/lib/server/upstream-auth";

export async function getCurrentSessionUser() {
  const sessionToken = await readAuthSessionToken();
  if (!sessionToken) {
    return null;
  }

  try {
    const session = await getUpstreamAuthSession(sessionToken);
    return session.authenticated ? session.user : null;
  } catch {
    return null;
  }
}
