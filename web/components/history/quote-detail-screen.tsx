"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ClipboardList, Download, LoaderCircle, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { formatCurrency, formatDateTime } from "@/lib/format";
import {
  formatQuoteFieldValue,
  getStepFields,
  quoteFieldConfigs,
  quoteSteps,
  toFormValues,
} from "@/lib/quote-form";
import { getAuthSession, getQuote, getQuoteReportUrl } from "@/lib/web-api";

export function QuoteDetailScreen() {
  const params = useParams<{ quoteId: string }>();
  const quoteId = params.quoteId;

  const quoteQuery = useQuery({
    queryKey: ["quote", quoteId],
    queryFn: ({ signal }) => getQuote(quoteId, signal),
    enabled: Boolean(quoteId),
  });
  const sessionQuery = useQuery({
    queryKey: ["auth-session"],
    queryFn: ({ signal }) => getAuthSession(signal),
  });

  if (quoteQuery.isPending) {
    return (
      <div className="flex h-[420px] items-center justify-center rounded-[30px] border border-dashed border-[var(--line)] bg-white/70 text-sm text-[var(--muted)]">
        <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
        Chargement du recapitulatif...
      </div>
    );
  }

  if (quoteQuery.isError || !quoteQuery.data) {
    return (
      <div className="rounded-[30px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-6 text-sm leading-7 text-[var(--danger)]">
        {quoteQuery.error instanceof Error
          ? quoteQuery.error.message
          : "Ce devis ne peut pas etre affiche pour le moment."}
      </div>
    );
  }

  const formValues = toFormValues(quoteQuery.data.input_payload);

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-3">
              <Badge className="border-white/20 bg-white/10 text-white">Recapitulatif</Badge>
              <div className="space-y-2">
                <h1 className="font-display text-4xl tracking-tight sm:text-5xl">Votre devis en un coup d&rsquo;oeil</h1>
                <p className="max-w-2xl text-sm leading-7 text-white/78 sm:text-base">
                  Retrouvez les informations transmises pour cette estimation, puis revenez au formulaire
                  si vous souhaitez ajuster votre devis.
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button asChild variant="secondary" className="border-white/20 bg-white/10 text-white hover:bg-white/16">
                <Link href="/mes-devis">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Retour a mes devis
                </Link>
              </Button>
              <Button asChild variant="secondary" className="border-white/20 bg-white/10 text-white hover:bg-white/16">
                <a href={getQuoteReportUrl(quoteId)}>
                  <Download className="mr-2 h-4 w-4" />
                  Telecharger le PDF
                </a>
              </Button>
              <Button asChild>
                <Link href={`/devis?quoteId=${quoteId}`}>Modifier ce devis</Link>
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="grid gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
            <div className="space-y-6">
              <div className="rounded-[28px] border border-[var(--line)] bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(242,228,214,0.9))] p-5">
                <p className="text-xs uppercase tracking-[0.2em] text-[var(--accent)]">Prime estimee</p>
                <p className="mt-3 font-display text-5xl text-[var(--foreground)]">
                  {formatCurrency(quoteQuery.data.result.prime_prediction)}
                </p>
                <div className="mt-4 space-y-2 text-sm text-[var(--muted)]">
                  <p>{formatDateTime(quoteQuery.data.created_at_utc)}</p>
                  <p>{formatQuoteFieldValue("type_contrat", quoteQuery.data.input_payload.type_contrat)}</p>
                  <p>
                    {formatQuoteFieldValue("marque_vehicule", quoteQuery.data.input_payload.marque_vehicule)}{" "}
                    {formatQuoteFieldValue("modele_vehicule", quoteQuery.data.input_payload.modele_vehicule)}
                  </p>
                </div>
              </div>

              <div className="rounded-[28px] border border-[var(--line)] bg-white/82 p-5">
                <div className="flex items-start gap-3">
                  <ShieldCheck className="mt-1 h-5 w-5 text-[var(--accent)]" />
                  <div>
                    <p className="font-semibold text-[var(--foreground)]">
                      {sessionQuery.data?.authenticated ? "Associe a votre compte" : "Disponible sur cet appareil"}
                    </p>
                    <p className="mt-1 text-sm leading-7 text-[var(--muted)]">
                      {sessionQuery.data?.authenticated
                        ? "Votre devis est rattache a votre compte. Vous pouvez le retrouver sur vos autres appareils."
                        : "Ce devis reste consultable ici et peut etre reutilise depuis votre historique."}
                    </p>
                  </div>
                </div>
              </div>

              <div className="rounded-[28px] border border-[var(--line)] bg-white/82 p-5">
                <p className="font-semibold text-[var(--foreground)]">Rapport PDF</p>
                <p className="mt-2 text-sm leading-7 text-[var(--muted)]">
                  Telechargez le rapport client de ce devis pour le conserver, l&rsquo;imprimer ou le partager
                  librement.
                </p>
                <div className="mt-4 flex flex-wrap gap-3">
                  <Button asChild variant="secondary">
                    <a href={getQuoteReportUrl(quoteId)}>
                      <Download className="mr-2 h-4 w-4" />
                      Telecharger le PDF
                    </a>
                  </Button>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              {quoteSteps.map((step) => (
                <Card key={step.id}>
                  <CardHeader className="pb-4">
                    <div className="flex items-center gap-3">
                      <div className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-[var(--surface-alt)] text-[var(--foreground)]">
                        <ClipboardList className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-[0.22em] text-[var(--accent)]">{step.eyebrow}</p>
                        <h2 className="font-display text-2xl text-[var(--foreground)]">{step.title}</h2>
                      </div>
                    </div>
                    <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{step.description}</p>
                  </CardHeader>
                  <CardContent className="grid gap-3 sm:grid-cols-2">
                    {getStepFields(step.id, formValues).map((fieldName) => (
                      <div
                        key={fieldName}
                        className="rounded-[22px] border border-[var(--line)] bg-white/76 p-4"
                      >
                        <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                          {quoteFieldConfigs[fieldName].label}
                        </p>
                        <p className="mt-2 text-sm font-semibold text-[var(--foreground)]">
                          {formatQuoteFieldValue(fieldName, quoteQuery.data.input_payload[fieldName])}
                        </p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
