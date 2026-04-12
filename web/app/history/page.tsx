import { redirect } from "next/navigation";
import { withAppBasePath } from "@/lib/app-paths";

export default function LegacyHistoryPage() {
  redirect(withAppBasePath("/mes-devis"));
}
