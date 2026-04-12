import { redirect } from "next/navigation";
import { withAppBasePath } from "@/lib/app-paths";

export default async function LegacyQuoteDetailPage({
  params,
}: {
  params: Promise<{ quoteId: string }>;
}) {
  const { quoteId } = await params;
  redirect(withAppBasePath(`/mes-devis/${quoteId}`));
}
