"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, LoaderCircle, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { loginAccount, registerAccount } from "@/lib/web-api";

type AuthFormProps = {
  mode: "login" | "register";
};

export function AuthForm({ mode }: AuthFormProps) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const authMutation = useMutation({
    mutationFn: async () =>
      mode === "login"
        ? loginAccount({ email, password })
        : registerAccount({ email, password }),
    onSuccess: async (session) => {
      await queryClient.invalidateQueries({ queryKey: ["auth-session"] });
      await queryClient.invalidateQueries({ queryKey: ["quotes"] });
      if (mode === "register" && session.email_verification_required) {
        router.push(`/confirmation-compte?email=${encodeURIComponent(email)}`);
        router.refresh();
        return;
      }
      if (session.user?.role === "admin") {
        router.push("/admin");
      } else {
        router.push("/compte");
      }
      router.refresh();
    },
  });

  const title = mode === "login" ? "Connexion a votre espace" : "Creez votre compte";
  const description =
    mode === "login"
      ? "Retrouvez vos devis, telechargez vos rapports PDF et gerez vos estimations depuis n'importe quelle machine."
      : "Creez votre espace, confirmez votre adresse email et retrouvez vos devis sur tous vos appareils.";
  const cta = mode === "login" ? "Se connecter" : "Creer mon compte";

  return (
    <div className="mx-auto max-w-2xl">
      <Card className="overflow-hidden">
        <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
          <div className="space-y-4">
            <Badge className="w-fit border-white/20 bg-white/10 text-white">
              {mode === "login" ? "Connexion" : "Inscription"}
            </Badge>
            <div className="space-y-3">
              <h1 className="font-display text-4xl tracking-tight sm:text-5xl">{title}</h1>
              <p className="max-w-2xl text-sm leading-7 text-white/78 sm:text-base">{description}</p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6 pt-6">
          {authMutation.isError ? (
            <div className="rounded-[24px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-4 text-sm leading-7 text-[var(--danger)]">
              {authMutation.error instanceof Error
                ? authMutation.error.message
                : "Une erreur est survenue pendant l'authentification."}
            </div>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-3 rounded-[24px] border border-[var(--line)] bg-white/82 p-4 sm:col-span-2">
              <div className="space-y-1">
                <span className="text-sm font-semibold text-[var(--foreground)]">Adresse email</span>
                <p className="text-xs leading-6 text-[var(--muted)]">
                  Cette adresse sera utilisee pour retrouver votre espace et recevoir vos rapports.
                </p>
              </div>
              <Input
                autoComplete="email"
                inputMode="email"
                placeholder="client@exemple.fr"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </label>

            <label className="space-y-3 rounded-[24px] border border-[var(--line)] bg-white/82 p-4 sm:col-span-2">
              <div className="space-y-1">
                <span className="text-sm font-semibold text-[var(--foreground)]">Mot de passe</span>
                <p className="text-xs leading-6 text-[var(--muted)]">
                  Choisissez un mot de passe de 8 caracteres minimum.
                </p>
              </div>
              <Input
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                type="password"
                placeholder="********"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </label>
          </div>

          <div className="flex flex-col gap-4 rounded-[26px] border border-[var(--line)] bg-[var(--surface-strong)] p-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <ShieldCheck className="mt-1 h-5 w-5 text-[var(--accent)]" />
              <p className="max-w-xl text-sm leading-7 text-[var(--muted)]">
                Votre compte vous permettra de retrouver vos devis, telecharger vos rapports PDF et,
                si besoin, reprendre une estimation depuis une autre machine.
              </p>
            </div>
            <Button
              type="button"
              disabled={authMutation.isPending || !email || password.length < 8}
              onClick={() => authMutation.mutate()}
            >
              {authMutation.isPending ? (
                <>
                  <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                  Traitement...
                </>
              ) : (
                <>
                  {cta}
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
