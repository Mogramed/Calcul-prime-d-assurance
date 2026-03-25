import { redirect } from "next/navigation";

import { HistoryScreen } from "@/components/history/history-screen";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function QuotesHistoryPage() {
  const user = await getCurrentSessionUser();

  if (!user) {
    redirect("/connexion");
  }

  return <HistoryScreen />;
}
