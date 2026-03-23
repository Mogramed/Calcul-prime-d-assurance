import { redirect } from "next/navigation";

import { AdminScreen } from "@/components/admin/admin-screen";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function AdminPage() {
  const user = await getCurrentSessionUser();

  if (!user) {
    redirect("/connexion");
  }

  if (user.role !== "admin") {
    redirect("/compte");
  }

  return <AdminScreen />;
}
