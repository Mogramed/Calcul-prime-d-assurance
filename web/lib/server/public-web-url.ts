function firstHeaderValue(value: string | null): string | null {
  if (!value) {
    return null;
  }

  const firstValue = value
    .split(",")
    .map((part) => part.trim())
    .find(Boolean);

  return firstValue ?? null;
}

function forwardedFieldValue(forwarded: string | null, fieldName: "host" | "proto"): string | null {
  const firstValue = firstHeaderValue(forwarded);
  if (!firstValue) {
    return null;
  }

  const match = firstValue.match(new RegExp(`${fieldName}=("[^"]+"|[^;]+)`, "i"));
  if (!match) {
    return null;
  }

  return match[1]?.replace(/^"|"$/g, "").trim() || null;
}

export function getPublicWebUrlFromRequest(request: Request, basePath = ""): string {
  const requestUrl = new URL(request.url);
  const forwarded = request.headers.get("forwarded");
  const proto =
    firstHeaderValue(request.headers.get("x-forwarded-proto")) ??
    forwardedFieldValue(forwarded, "proto") ??
    requestUrl.protocol.replace(/:$/, "");
  const host =
    firstHeaderValue(request.headers.get("x-forwarded-host")) ??
    forwardedFieldValue(forwarded, "host") ??
    firstHeaderValue(request.headers.get("host")) ??
    requestUrl.host;

  const normalizedBasePath = basePath.trim().replace(/\/$/, "");
  return `${proto}://${host}${normalizedBasePath}`;
}
