"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, LoaderCircle, RefreshCcw, ShieldCheck } from "lucide-react";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { formatCurrency, formatDateTime } from "@/lib/format";
import { formatQuoteFieldValue } from "@/lib/quote-form";
import { getAuthSession, listQuotes } from "@/lib/web-api";

export function HistoryScreen() {
  const sessionQuery = useQuery({
    queryKey: ["auth-session"],
    queryFn: ({ signal }) => getAuthSession(signal),
  });
  const historyQuery = useQuery({
    queryKey: ["quotes"],
    queryFn: ({ signal }) => listQuotes(signal),
  });

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-3">
              <Badge className="border-white/20 bg-white/10 text-white">Mes devis</Badge>
              <div className="space-y-2">
                <h1 className="font-display text-4xl tracking-tight sm:text-5xl">Retrouvez vos derniers devis</h1>
                <p className="max-w-2xl text-sm leading-7 text-white/78 sm:text-base">
                  {sessionQuery.data?.authenticated
                    ? "Vos devis sont rattaches a votre compte. Vous pouvez les consulter, les modifier ou relancer une nouvelle estimation depuis n'importe quelle machine."
                    : "Vos derniers devis restent disponibles sur cet appareil. Vous pouvez les consulter, les modifier ou relancer une nouvelle estimation."}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button asChild variant="secondary" className="border-white/20 bg-white/10 text-white hover:bg-white/16">
                <Link href="/devis">Demarrer un devis</Link>
              </Button>
              <Button
                variant="ghost"
                className="text-white hover:bg-white/10"
                onClick={() => historyQuery.refetch()}
              >
                <RefreshCcw className="mr-2 h-4 w-4" />
                Actualiser
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          {historyQuery.isPending ? (
            <div className="flex h-48 items-center justify-center rounded-[26px] border border-dashed border-[var(--line)] bg-white/70 text-sm text-[var(--muted)]">
              <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
              Chargement de vos devis...
            </div>
          ) : historyQuery.isError ? (
            <div className="rounded-[26px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-5 text-sm leading-7 text-[var(--danger)]">
              {historyQuery.error instanceof Error
                ? historyQuery.error.message
                : "Vos devis ne peuvent pas etre affiches pour le moment."}
            </div>
          ) : historyQuery.data?.quotes.length ? (
            <div className="grid gap-4">
              {historyQuery.data.quotes.map((quote) => (
                <div
                  key={quote.id}
                  className="grid gap-4 rounded-[28px] border border-[var(--line)] bg-white/82 p-5 lg:grid-cols-[1fr_auto]"
                >
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center gap-3 text-xs text-[var(--muted)]">
                      <span className="rounded-full bg-[var(--surface-alt)] px-3 py-1">
                        {formatDateTime(quote.created_at_utc)}
                      </span>
                      <span className="rounded-full bg-[var(--surface-alt)] px-3 py-1">
                        {formatQuoteFieldValue("type_contrat", quote.type_contrat)}
                      </span>
                    </div>
                    <div className="space-y-2">
                      <p className="text-xs uppercase tracking-[0.2em] text-[var(--accent)]">Prime estimee</p>
                      <div className="flex flex-wrap items-end justify-between gap-4">
                        <div>
                          <p className="font-display text-4xl text-[var(--foreground)]">
                            {formatCurrency(quote.prime_prediction)}
                          </p>
                          <p className="mt-2 text-sm leading-7 text-[var(--muted)]">
                            {formatQuoteFieldValue("marque_vehicule", quote.marque_vehicule)}{" "}
                            {formatQuoteFieldValue("modele_vehicule", quote.modele_vehicule)}
                          </p>
                        </div>
                        <div className="rounded-[22px] border border-[var(--line)] bg-[var(--surface-alt)]/70 px-4 py-3 text-sm text-[var(--muted)]">
                          {sessionQuery.data?.authenticated
                            ? "Ce devis est rattache a votre compte."
                            : "Ce devis reste disponible sur cet appareil."}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-3 lg:flex-col">
                    <Button asChild variant="secondary">
                      <Link href={`/devis?quoteId=${quote.id}`}>Modifier</Link>
                    </Button>
                    <Button asChild>
                      <Link href={`/mes-devis/${quote.id}`}>
                        Voir le recapitulatif
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Link>
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-[28px] border border-dashed border-[var(--line)] bg-white/70 p-8">
              <div className="flex items-start gap-4">
                <ShieldCheck className="mt-1 h-5 w-5 text-[var(--accent)]" />
                <div className="space-y-3">
                  <h2 className="font-display text-3xl text-[var(--foreground)]">Aucun devis pour le moment</h2>
                  <p className="max-w-2xl text-sm leading-7 text-[var(--muted)]">
                    Lancez votre premiere estimation pour commencer votre historique sur cet appareil.
                  </p>
                  <Button asChild>
                    <Link href="/devis">Demarrer un devis</Link>
                  </Button>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
