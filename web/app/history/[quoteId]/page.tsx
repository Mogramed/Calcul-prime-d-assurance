import { redirect } from "next/navigation";

export default async function LegacyQuoteDetailPage({
  params,
}: {
  params: Promise<{ quoteId: string }>;
}) {
  const { quoteId } = await params;
  redirect(`/mes-devis/${quoteId}`);
}
