"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, LoaderCircle, MailWarning } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { verifyAccountEmail } from "@/lib/web-api";

export default function VerifyEmailPage() {
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const token = searchParams.get("token")?.trim() ?? "";
  const hasTriggered = useRef(false);

  const verificationMutation = useMutation({
    mutationFn: async (verificationToken: string) =>
      verifyAccountEmail({ token: verificationToken }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["auth-session"] });
    },
  });

  useEffect(() => {
    if (!token || hasTriggered.current) {
      return;
    }
    hasTriggered.current = true;
    verificationMutation.mutate(token);
  }, [token, verificationMutation]);

  return (
    <div className="mx-auto max-w-3xl">
      <Card className="overflow-hidden">
        <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
          <div className="space-y-4">
            <Badge className="w-fit border-white/20 bg-white/10 text-white">Verification du compte</Badge>
            <div className="space-y-3">
              <h1 className="font-display text-4xl tracking-tight sm:text-5xl">
                Confirmation de votre adresse email
              </h1>
              <p className="max-w-2xl text-sm leading-7 text-white/78 sm:text-base">
                Nous verifions votre lien pour finaliser l'activation de votre espace Nova Assurances.
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-6">
          {!token ? (
            <div className="rounded-[24px] border border-[color:color-mix(in_srgb,var(--warning)_22%,white)] bg-[color:color-mix(in_srgb,var(--warning)_8%,white)] p-5 text-sm leading-7 text-[var(--foreground)]">
              Le lien de verification est incomplet. Ouvrez a nouveau l'email recu apres votre inscription.
            </div>
          ) : verificationMutation.isPending ? (
            <div className="flex min-h-52 items-center justify-center rounded-[24px] border border-dashed border-[var(--line)] bg-white/72 text-sm text-[var(--muted)]">
              <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
              Verification en cours...
            </div>
          ) : verificationMutation.isSuccess ? (
            <div className="space-y-5">
              <div className="rounded-[24px] border border-[color:color-mix(in_srgb,var(--success)_18%,white)] bg-[color:color-mix(in_srgb,var(--success)_8%,white)] p-5 text-sm leading-7 text-[var(--foreground)]">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="mt-1 h-5 w-5 text-[var(--success)]" />
                  <div>
                    <p className="font-semibold">Adresse email confirmee</p>
                    <p className="mt-1">
                      Votre compte est maintenant valide. Vous pouvez reprendre vos devis ou vous connecter depuis n'importe quelle machine.
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                <Button asChild>
                  <Link href="/compte">Acceder a mon compte</Link>
                </Button>
                <Button asChild variant="secondary">
                  <Link href="/connexion">Me connecter</Link>
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              <div className="rounded-[24px] border border-[color:color-mix(in_srgb,var(--danger)_18%,white)] bg-[color:color-mix(in_srgb,var(--danger)_8%,white)] p-5 text-sm leading-7 text-[var(--foreground)]">
                <div className="flex items-start gap-3">
                  <MailWarning className="mt-1 h-5 w-5 text-[var(--danger)]" />
                  <div>
                    <p className="font-semibold">Le lien n'est plus valide</p>
                    <p className="mt-1">
                      {verificationMutation.error instanceof Error
                        ? verificationMutation.error.message
                        : "Ce lien de verification est invalide ou a deja ete utilise."}
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                <Button asChild>
                  <Link href="/connexion">Aller a la connexion</Link>
                </Button>
                <Button asChild variant="secondary">
                  <Link href="/inscription">Creer un autre compte</Link>
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
