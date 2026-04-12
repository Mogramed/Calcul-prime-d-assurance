import { redirect } from "next/navigation";

function buildSearchQuery(
  searchParams: Record<string, string | string[] | undefined>,
) {
  const query = new URLSearchParams();

  for (const [key, value] of Object.entries(searchParams)) {
    if (Array.isArray(value)) {
      for (const item of value) {
        query.append(key, item);
      }
      continue;
    }

    if (value) {
      query.set(key, value);
    }
  }

  return query.toString();
}

export default async function LegacyQuotePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const resolvedSearchParams = await searchParams;
  const query = buildSearchQuery(resolvedSearchParams);

  redirect(query ? `/devis?${query}` : "/devis");
}
