import Link from "next/link";
import { redirect } from "next/navigation";

import { LogoutButton } from "@/components/auth/logout-button";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { formatDateTime } from "@/lib/format";
import { getCurrentSessionUser } from "@/lib/server/current-user";

export default async function AccountPage() {
  const user = await getCurrentSessionUser();

  if (!user) {
    redirect("/connexion");
  }

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
          <div className="space-y-4">
            <Badge className="w-fit border-white/20 bg-white/10 text-white">Mon compte</Badge>
            <div className="space-y-3">
              <h1 className="font-display text-4xl tracking-tight sm:text-5xl">Votre espace Nova Assurances</h1>
              <p className="max-w-3xl text-sm leading-7 text-white/78 sm:text-base">
                Retrouvez vos devis sur tous vos appareils, telechargez vos rapports PDF et envoyez-les
                directement sur votre adresse email.
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="grid gap-6 pt-6 lg:grid-cols-[minmax(0,1fr)_auto]">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-[24px] border border-[var(--line)] bg-white/82 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Adresse email</p>
              <p className="mt-2 font-semibold text-[var(--foreground)]">{user.email}</p>
            </div>
            <div className="rounded-[24px] border border-[var(--line)] bg-white/82 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Role</p>
              <p className="mt-2 font-semibold text-[var(--foreground)]">
                {user.role === "admin" ? "Administrateur" : "Client"}
              </p>
            </div>
            <div className="rounded-[24px] border border-[var(--line)] bg-white/82 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Compte cree le</p>
              <p className="mt-2 font-semibold text-[var(--foreground)]">{formatDateTime(user.created_at_utc)}</p>
            </div>
            <div className="rounded-[24px] border border-[var(--line)] bg-white/82 p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Statut</p>
              <p className="mt-2 font-semibold text-[var(--foreground)]">{user.is_active ? "Actif" : "Inactif"}</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-3 lg:flex-col">
            <Button asChild>
              <Link href="/mes-devis">Voir mes devis</Link>
            </Button>
            <Button asChild variant="secondary">
              <Link href="/devis">Nouveau devis</Link>
            </Button>
            {user.role === "admin" ? (
              <Button asChild variant="secondary">
                <Link href="/admin">Console admin</Link>
              </Button>
            ) : null}
            <LogoutButton />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
