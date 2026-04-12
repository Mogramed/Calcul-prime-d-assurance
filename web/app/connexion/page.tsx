import { redirect } from "next/navigation";

import { AuthForm } from "@/components/auth/auth-form";
import { withAppBasePath } from "@/lib/app-paths";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function LoginPage() {
  const user = await getCurrentSessionUser();

  if (user) {
    redirect(withAppBasePath(user.role === "admin" ? "/admin" : "/compte"));
  }

  return <AuthForm mode="login" />;
}
