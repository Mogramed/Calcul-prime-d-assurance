"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LoaderCircle, ShieldCheck, Trash2, UsersRound } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { formatCurrency, formatDateTime } from "@/lib/format";
import {
  deactivateAdminUser,
  deleteAdminQuote,
  listAdminQuotes,
  listAdminUsers,
} from "@/lib/web-api";

export function AdminScreen() {
  const queryClient = useQueryClient();
  const usersQuery = useQuery({
    queryKey: ["admin-users"],
    queryFn: ({ signal }) => listAdminUsers(signal),
  });
  const quotesQuery = useQuery({
    queryKey: ["admin-quotes"],
    queryFn: ({ signal }) => listAdminQuotes(signal),
  });

  const userMutation = useMutation({
    mutationFn: deactivateAdminUser,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["admin-users"] });
    },
  });

  const quoteMutation = useMutation({
    mutationFn: deleteAdminQuote,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["admin-quotes"] });
      await queryClient.invalidateQueries({ queryKey: ["quotes"] });
    },
  });

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
          <div className="space-y-4">
            <Badge className="w-fit border-white/20 bg-white/10 text-white">Console admin</Badge>
            <div className="space-y-3">
              <h1 className="font-display text-4xl tracking-tight sm:text-5xl">Pilotage des comptes et des devis</h1>
              <p className="max-w-3xl text-sm leading-7 text-white/78 sm:text-base">
                Cette zone est reservee aux comptes administrateurs. Les suppressions sont appliquees
                cote serveur et n&rsquo;apparaissent jamais comme de simples restrictions d&rsquo;interface.
              </p>
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <UsersRound className="h-5 w-5 text-[var(--accent)]" />
              <div>
                <h2 className="font-display text-2xl text-[var(--foreground)]">Comptes utilisateurs</h2>
                <p className="text-sm leading-7 text-[var(--muted)]">Activer ou desactiver les comptes clients et admin.</p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {usersQuery.isPending ? (
              <div className="flex h-36 items-center justify-center rounded-[24px] border border-dashed border-[var(--line)] bg-white/70 text-sm text-[var(--muted)]">
                <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                Chargement des comptes...
              </div>
            ) : usersQuery.isError ? (
              <div className="rounded-[24px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-4 text-sm leading-7 text-[var(--danger)]">
                {usersQuery.error instanceof Error ? usersQuery.error.message : "La liste des comptes est indisponible."}
              </div>
            ) : (
              usersQuery.data?.users.map((user) => (
                <div key={user.id} className="rounded-[24px] border border-[var(--line)] bg-white/82 p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold text-[var(--foreground)]">{user.email}</p>
                      <p className="mt-1 text-sm text-[var(--muted)]">
                        Role: {user.role === "admin" ? "Administrateur" : "Client"} -{" "}
                        {user.is_active ? "Actif" : "Inactif"}
                      </p>
                      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                        Cree le {formatDateTime(user.created_at_utc)}
                      </p>
                    </div>
                    {user.is_active ? (
                      <Button
                        type="button"
                        variant="secondary"
                        disabled={userMutation.isPending}
                        onClick={() => userMutation.mutate(user.id)}
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Desactiver
                      </Button>
                    ) : (
                      <Badge>Inactif</Badge>
                    )}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <ShieldCheck className="h-5 w-5 text-[var(--accent)]" />
              <div>
                <h2 className="font-display text-2xl text-[var(--foreground)]">Devis recents</h2>
                <p className="text-sm leading-7 text-[var(--muted)]">Superviser les devis et supprimer ceux qui ne doivent plus apparaitre.</p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {quotesQuery.isPending ? (
              <div className="flex h-36 items-center justify-center rounded-[24px] border border-dashed border-[var(--line)] bg-white/70 text-sm text-[var(--muted)]">
                <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                Chargement des devis...
              </div>
            ) : quotesQuery.isError ? (
              <div className="rounded-[24px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-4 text-sm leading-7 text-[var(--danger)]">
                {quotesQuery.error instanceof Error ? quotesQuery.error.message : "La liste des devis est indisponible."}
              </div>
            ) : (
              quotesQuery.data?.quotes.map((quote) => (
                <div key={quote.id} className="rounded-[24px] border border-[var(--line)] bg-white/82 p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold text-[var(--foreground)]">
                        {quote.marque_vehicule} {quote.modele_vehicule}
                      </p>
                      <p className="mt-1 text-sm text-[var(--muted)]">
                        {formatCurrency(quote.prime_prediction)} - {quote.type_contrat}
                      </p>
                      <p className="mt-1 text-sm text-[var(--muted)]">
                        Proprietaire: {quote.owner_email ?? "Visiteur anonyme"}
                      </p>
                      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                        {formatDateTime(quote.created_at_utc)}
                      </p>
                    </div>
                    {quote.deleted_at_utc ? (
                      <Badge>Supprime</Badge>
                    ) : (
                      <Button
                        type="button"
                        variant="secondary"
                        disabled={quoteMutation.isPending}
                        onClick={() => quoteMutation.mutate(quote.id)}
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Supprimer
                      </Button>
                    )}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
