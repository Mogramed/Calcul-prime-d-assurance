import { Suspense } from "react";

import { QuoteWorkbench } from "@/components/quote/quote-workbench";

export default function QuotePage() {
  return (
    <Suspense fallback={null}>
      <QuoteWorkbench />
    </Suspense>
  );
}
