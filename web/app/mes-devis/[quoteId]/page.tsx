import { redirect } from "next/navigation";

import { QuoteDetailScreen } from "@/components/history/quote-detail-screen";
import { withAppBasePath } from "@/lib/app-paths";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function QuoteDetailPage() {
  const user = await getCurrentSessionUser();

  if (!user) {
    redirect(withAppBasePath("/connexion"));
  }

  return <QuoteDetailScreen />;
}
