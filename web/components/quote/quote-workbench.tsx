"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  ArrowRight,
  CarFront,
  Check,
  ClipboardList,
  LoaderCircle,
  ShieldCheck,
  Sparkles,
  UsersRound,
} from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { formatCurrency, formatDateTime } from "@/lib/format";
import {
  defaultQuoteValues,
  formatQuoteFieldValue,
  getFieldOptions,
  getStepFields,
  isSecondDriverEnabled,
  quoteFieldConfigs,
  quoteFormSchema,
  quoteSteps,
  toFormValues,
  toPredictionInput,
  type QuoteFormValues,
  type QuoteStepId,
} from "@/lib/quote-form";
import { createQuote, getAuthSession, getQuote } from "@/lib/web-api";

const stepIcons: Record<QuoteStepId, typeof ClipboardList> = {
  profile: ClipboardList,
  drivers: UsersRound,
  vehicle: CarFront,
};

export function QuoteWorkbench() {
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const quoteId = searchParams.get("quoteId");
  const hasQuoteId = Boolean(quoteId);
  const [activeStepIndex, setActiveStepIndex] = useState(0);

  const form = useForm<QuoteFormValues>({
    resolver: zodResolver(quoteFormSchema),
    defaultValues: defaultQuoteValues,
  });

  const formValues = form.watch();
  const currentStep = quoteSteps[activeStepIndex];
  const currentStepFields = getStepFields(currentStep.id, formValues);
  const secondDriverEnabled = isSecondDriverEnabled(formValues);
  const modelOptions = getFieldOptions("modele_vehicule", formValues);

  const quoteQuery = useQuery({
    queryKey: ["quote", quoteId],
    queryFn: ({ signal }) => getQuote(quoteId as string, signal),
    enabled: hasQuoteId,
  });
  const authSessionQuery = useQuery({
    queryKey: ["auth-session"],
    queryFn: ({ signal }) => getAuthSession(signal),
  });

  useEffect(() => {
    if (quoteQuery.data) {
      form.reset(toFormValues(quoteQuery.data.input_payload));
    }
  }, [form, quoteQuery.data]);

  useEffect(() => {
    if (formValues.modele_vehicule && !modelOptions.some((option) => option.value === formValues.modele_vehicule)) {
      form.setValue("modele_vehicule", "");
    }
  }, [form, formValues.modele_vehicule, modelOptions]);

  useEffect(() => {
    if (!secondDriverEnabled) {
      if (formValues.age_conducteur2) {
        form.setValue("age_conducteur2", "");
      }
      if (formValues.sex_conducteur2) {
        form.setValue("sex_conducteur2", "");
      }
      if (formValues.anciennete_permis2) {
        form.setValue("anciennete_permis2", "");
      }
    }
  }, [
    form,
    formValues.age_conducteur2,
    formValues.anciennete_permis2,
    formValues.sex_conducteur2,
    secondDriverEnabled,
  ]);

  const createQuoteMutation = useMutation({
    mutationFn: async (values: QuoteFormValues) => createQuote(toPredictionInput(values)),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["quotes"] });
      setActiveStepIndex(quoteSteps.length - 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
  });

  const activeQuote = createQuoteMutation.data ?? quoteQuery.data;

  async function goToNextStep() {
    const stepIsValid = await form.trigger(currentStepFields, { shouldFocus: true });
    if (stepIsValid) {
      setActiveStepIndex((currentIndex) => Math.min(currentIndex + 1, quoteSteps.length - 1));
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_390px]">
      <div className="space-y-6">
        <Card className="overflow-hidden">
          <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
            <div className="flex flex-wrap items-start justify-between gap-6">
              <div className="space-y-4">
                <Badge className="border-white/20 bg-white/10 text-white">Devis auto Nova Assurances</Badge>
                <div className="space-y-3">
                  <h1 className="max-w-4xl font-display text-4xl tracking-tight sm:text-5xl">
                    Obtenez une estimation claire de votre prime auto, etape par etape.
                  </h1>
                  <p className="max-w-2xl text-sm leading-7 text-white/76 sm:text-base">
                    Nous vous guidons dans la saisie pour aller a l'essentiel, sans jargon
                    inutile. Votre estimation apparaitra des que le dossier est complet.
                  </p>
                </div>
              </div>

              <div className="grid gap-3 text-sm text-white/78 sm:grid-cols-3 lg:max-w-xs lg:grid-cols-1">
                <div className="rounded-[26px] border border-white/12 bg-white/10 px-4 py-3">
                  <p className="font-semibold text-white">Rapide</p>
                  <p className="mt-1 text-xs leading-6">Un parcours guide en trois etapes seulement.</p>
                </div>
                <div className="rounded-[26px] border border-white/12 bg-white/10 px-4 py-3">
                  <p className="font-semibold text-white">Sans engagement</p>
                  <p className="mt-1 text-xs leading-6">Une estimation indicative, disponible en quelques instants.</p>
                </div>
                <div className="rounded-[26px] border border-white/12 bg-white/10 px-4 py-3">
                  <p className="font-semibold text-white">Accessible</p>
                  <p className="mt-1 text-xs leading-6">
                    {authSessionQuery.data?.authenticated
                      ? "Vos devis suivent votre compte, meme sur un autre appareil."
                      : "Vos derniers devis restent consultables sur cet appareil."}
                  </p>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid gap-3 md:grid-cols-3">
              {quoteSteps.map((step, index) => {
                const StepIcon = stepIcons[step.id];
                const isActive = index === activeStepIndex;
                const isDone = index < activeStepIndex;

                return (
                  <button
                    key={step.id}
                    type="button"
                    className={`rounded-[26px] border px-4 py-4 text-left transition ${
                      isActive
                        ? "border-[var(--accent)] bg-[var(--surface-strong)] shadow-[0_20px_45px_rgba(20,33,50,0.08)]"
                        : isDone
                          ? "border-[var(--line)] bg-white/88"
                          : "border-[var(--line)] bg-[var(--surface-alt)]/70"
                    }`}
                    onClick={() => {
                      if (index <= activeStepIndex) {
                        setActiveStepIndex(index);
                      }
                    }}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-[var(--surface-alt)] text-[var(--foreground)]">
                        {isDone ? <Check className="h-5 w-5 text-[var(--accent)]" /> : <StepIcon className="h-5 w-5" />}
                      </div>
                      <span className="text-xs uppercase tracking-[0.22em] text-[var(--muted)]">
                        {step.eyebrow}
                      </span>
                    </div>
                    <h2 className="mt-4 font-display text-2xl text-[var(--foreground)]">{step.title}</h2>
                    <p className="mt-2 text-sm leading-7 text-[var(--muted)]">{step.description}</p>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card className="overflow-hidden">
          <CardHeader className="border-b border-[var(--line)] pb-5">
            <div className="space-y-2">
              <Badge>{currentStep.eyebrow}</Badge>
              <h2 className="font-display text-3xl text-[var(--foreground)]">{currentStep.title}</h2>
              <p className="max-w-2xl text-sm leading-7 text-[var(--muted)]">{currentStep.description}</p>
            </div>
          </CardHeader>
          <CardContent className="pt-6">
            {hasQuoteId && quoteQuery.isPending ? (
              <div className="flex h-48 items-center justify-center rounded-[26px] border border-dashed border-[var(--line)] bg-white/70 text-sm text-[var(--muted)]">
                <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                Recuperation de votre devis en cours...
              </div>
            ) : hasQuoteId && quoteQuery.isError ? (
              <div className="rounded-[26px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-5 text-sm leading-7 text-[var(--danger)]">
                {quoteQuery.error instanceof Error
                  ? quoteQuery.error.message
                  : "Ce devis ne peut pas etre charge pour le moment."}
              </div>
            ) : (
              <form
                className="space-y-8"
                onSubmit={form.handleSubmit((values) => createQuoteMutation.mutate(values))}
              >
                <div className="grid gap-4 sm:grid-cols-2">
                  {currentStepFields.map((fieldName) => {
                    const fieldConfig = quoteFieldConfigs[fieldName];
                    const options = getFieldOptions(fieldName, formValues);
                    const error = form.formState.errors[fieldName]?.message;
                    const isSelectField = fieldConfig.kind === "select";
                    const isModelField = fieldName === "modele_vehicule";

                    return (
                      <label
                        key={fieldName}
                        className="space-y-3 rounded-[24px] border border-[var(--line)] bg-white/82 p-4"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center justify-between gap-4">
                            <span className="text-sm font-semibold text-[var(--foreground)]">{fieldConfig.label}</span>
                            {error ? (
                              <span className="text-xs font-medium text-[var(--danger)]">{error}</span>
                            ) : null}
                          </div>
                          <p className="text-xs leading-6 text-[var(--muted)]">{fieldConfig.description}</p>
                        </div>

                        {isSelectField ? (
                          <Select
                            disabled={isModelField && !formValues.marque_vehicule}
                            {...form.register(fieldName)}
                          >
                            <option value="">
                              {isModelField && !formValues.marque_vehicule
                                ? "Choisissez d'abord une marque"
                                : "Selectionnez une option"}
                            </option>
                            {options.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </Select>
                        ) : (
                          <Input
                            type={fieldConfig.inputMode === "decimal" || fieldConfig.inputMode === "numeric" ? "number" : "text"}
                            inputMode={fieldConfig.inputMode}
                            step={fieldConfig.inputMode === "decimal" ? "any" : undefined}
                            placeholder={fieldConfig.placeholder}
                            {...form.register(fieldName)}
                          />
                        )}
                      </label>
                    );
                  })}
                </div>

                <div className="flex flex-col gap-4 rounded-[26px] border border-[var(--line)] bg-[var(--surface-strong)] p-5 sm:flex-row sm:items-center sm:justify-between">
                  <div className="space-y-2">
                    <p className="text-sm font-semibold text-[var(--foreground)]">Une estimation simple et lisible</p>
                    <p className="max-w-xl text-sm leading-7 text-[var(--muted)]">
                      {authSessionQuery.data?.authenticated
                        ? "Chaque devis est rattache a votre compte pour le retrouver plus tard et le modifier depuis n'importe quelle machine."
                        : "Chaque devis enregistre votre saisie sur cet appareil pour que vous puissiez le retrouver plus tard et le modifier si besoin."}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    {form.formState.isDirty ? (
                      <Button
                        type="button"
                        variant="secondary"
                        onClick={() => {
                          form.reset(defaultQuoteValues);
                          setActiveStepIndex(0);
                        }}
                      >
                        Reinitialiser
                      </Button>
                    ) : null}
                    {activeStepIndex > 0 ? (
                      <Button
                        type="button"
                        variant="ghost"
                        onClick={() => setActiveStepIndex((index) => Math.max(index - 1, 0))}
                      >
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Etape precedente
                      </Button>
                    ) : null}
                    {activeStepIndex < quoteSteps.length - 1 ? (
                      <Button type="button" onClick={goToNextStep}>
                        Continuer
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                    ) : (
                      <Button type="submit" disabled={createQuoteMutation.isPending}>
                        {createQuoteMutation.isPending ? (
                          <>
                            <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                            Estimation en cours
                          </>
                        ) : (
                          <>
                            Obtenir mon estimation
                            <Sparkles className="ml-2 h-4 w-4" />
                          </>
                        )}
                      </Button>
                    )}
                  </div>
                </div>
              </form>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="space-y-6 lg:sticky lg:top-24 lg:self-start">
        <Card className="overflow-hidden">
          <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(180deg,rgba(247,240,232,0.96),rgba(255,255,255,0.74))] pb-5">
            <Badge className="w-fit">Votre estimation</Badge>
            <h2 className="mt-2 font-display text-3xl text-[var(--foreground)]">Restez concentre sur l'essentiel</h2>
            <p className="mt-2 text-sm leading-7 text-[var(--muted)]">
              Nous mettons la prime estimee au premier plan, avec un recapitulatif simple de votre dossier.
            </p>
          </CardHeader>
          <CardContent className="space-y-5 pt-6">
            {createQuoteMutation.isError ? (
              <div className="rounded-[24px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-4 text-sm leading-7 text-[var(--danger)]">
                {createQuoteMutation.error instanceof Error
                  ? createQuoteMutation.error.message
                  : "Une erreur est survenue pendant le calcul."}
              </div>
            ) : null}

            {activeQuote ? (
              <div className="space-y-5">
                <div className="rounded-[28px] border border-[var(--line)] bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(242,228,214,0.9))] p-5">
                  <p className="text-xs uppercase tracking-[0.24em] text-[var(--accent)]">Prime estimee</p>
                  <p className="mt-3 font-display text-5xl text-[var(--foreground)]">
                    {formatCurrency(activeQuote.result.prime_prediction)}
                  </p>
                  <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
                    Estimation indicative, sans engagement, calculee a partir des informations renseignees.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2 text-xs text-[var(--muted)]">
                    <span className="rounded-full bg-white/80 px-3 py-1">
                      {formatQuoteFieldValue("type_contrat", activeQuote.input_payload.type_contrat)}
                    </span>
                    <span className="rounded-full bg-white/80 px-3 py-1">
                      {formatDateTime(activeQuote.created_at_utc)}
                    </span>
                  </div>
                </div>

                <div className="grid gap-3">
                  <div className="rounded-[24px] border border-[var(--line)] bg-white/80 px-4 py-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">Vehicule</p>
                    <p className="mt-2 font-semibold text-[var(--foreground)]">
                      {formatQuoteFieldValue("marque_vehicule", activeQuote.input_payload.marque_vehicule)}{" "}
                      {formatQuoteFieldValue("modele_vehicule", activeQuote.input_payload.modele_vehicule)}
                    </p>
                  </div>
                  <div className="rounded-[24px] border border-[var(--line)] bg-white/80 px-4 py-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">Profil</p>
                    <p className="mt-2 font-semibold text-[var(--foreground)]">
                      {formatQuoteFieldValue("utilisation", activeQuote.input_payload.utilisation)}
                    </p>
                    <p className="mt-1 text-sm text-[var(--muted)]">
                      Conducteur principal: {formatQuoteFieldValue("age_conducteur1", activeQuote.input_payload.age_conducteur1)} ans
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button asChild variant="secondary">
                    <Link href="/mes-devis">Voir mes devis</Link>
                  </Button>
                  <Button asChild>
                    <Link href={`/mes-devis/${activeQuote.id}`}>Voir le recapitulatif</Link>
                  </Button>
                </div>
              </div>
            ) : (
              <div className="rounded-[28px] border border-dashed border-[var(--line)] bg-white/72 p-6 text-sm leading-7 text-[var(--muted)]">
                Votre estimation apparaitra ici une fois les trois etapes completees.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-4">
            <h2 className="font-display text-2xl text-[var(--foreground)]">Recapitulatif rapide</h2>
            <p className="text-sm leading-7 text-[var(--muted)]">
              Un resume de votre dossier pour suivre l'avancement sans quitter le formulaire.
            </p>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-[var(--muted)]">
            <div className="rounded-2xl border border-[var(--line)] bg-white/75 px-4 py-3">
              <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Formule</p>
              <p className="mt-2 font-semibold text-[var(--foreground)]">
                {formatQuoteFieldValue("type_contrat", formValues.type_contrat)}
              </p>
            </div>
            <div className="rounded-2xl border border-[var(--line)] bg-white/75 px-4 py-3">
              <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Conducteurs</p>
              <p className="mt-2 font-semibold text-[var(--foreground)]">
                Principal: {formValues.age_conducteur1 ? `${formValues.age_conducteur1} ans` : "A renseigner"}
              </p>
              <p className="mt-1 text-sm text-[var(--muted)]">
                Deuxieme conducteur: {formatQuoteFieldValue("conducteur2", formValues.conducteur2)}
              </p>
            </div>
            <div className="rounded-2xl border border-[var(--line)] bg-white/75 px-4 py-3">
              <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Vehicule</p>
              <p className="mt-2 font-semibold text-[var(--foreground)]">
                {formValues.marque_vehicule
                  ? formatQuoteFieldValue("marque_vehicule", formValues.marque_vehicule)
                  : "Marque a choisir"}
              </p>
              <p className="mt-1 text-sm text-[var(--muted)]">
                {formValues.modele_vehicule
                  ? formatQuoteFieldValue("modele_vehicule", formValues.modele_vehicule)
                  : "Modele a choisir"}
              </p>
            </div>
            <div className="rounded-2xl border border-[var(--line)] bg-[var(--surface-alt)]/70 px-4 py-3">
              <div className="flex items-start gap-3">
                <ShieldCheck className="mt-0.5 h-4 w-4 text-[var(--accent)]" />
                <p className="leading-7">
                  {authSessionQuery.data?.authenticated
                    ? "Vos devis restent disponibles dans votre compte."
                    : "Vos derniers devis restent disponibles sur cet appareil."}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
