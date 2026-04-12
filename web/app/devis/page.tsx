import { Suspense } from "react";
import { redirect } from "next/navigation";

import { QuoteWorkbench } from "@/components/quote/quote-workbench";
import { withAppBasePath } from "@/lib/app-paths";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function QuotePage() {
  const user = await getCurrentSessionUser();

  if (!user) {
    redirect(withAppBasePath("/connexion"));
  }

  return (
    <Suspense fallback={null}>
      <QuoteWorkbench />
    </Suspense>
  );
}
