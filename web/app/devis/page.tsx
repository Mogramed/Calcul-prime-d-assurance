import { Suspense } from "react";
import { redirect } from "next/navigation";

import { QuoteWorkbench } from "@/components/quote/quote-workbench";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function QuotePage() {
  const user = await getCurrentSessionUser();

  if (!user) {
    redirect("/connexion");
  }

  return (
    <Suspense fallback={null}>
      <QuoteWorkbench />
    </Suspense>
  );
}
