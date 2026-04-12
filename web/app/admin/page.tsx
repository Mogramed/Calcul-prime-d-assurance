import { redirect } from "next/navigation";

import { AdminScreen } from "@/components/admin/admin-screen";
import { withAppBasePath } from "@/lib/app-paths";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function AdminPage() {
  const user = await getCurrentSessionUser();

  if (!user) {
    redirect(withAppBasePath("/connexion"));
  }

  if (user.role !== "admin") {
    redirect(withAppBasePath("/compte"));
  }

  return <AdminScreen />;
}
