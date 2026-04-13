import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

type AccountConfirmationPageProps = {
  searchParams: Promise<{
    email?: string;
  }>;
};

export default async function AccountConfirmationPage({
  searchParams,
}: AccountConfirmationPageProps) {
  const { email } = await searchParams;

  return (
    <div className="mx-auto max-w-3xl">
      <Card className="overflow-hidden">
        <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.96),rgba(26,40,60,0.9))] pb-8 text-white">
          <div className="space-y-4">
            <Badge className="w-fit border-white/20 bg-white/10 text-white">
              Confirmation requise
            </Badge>
            <div className="space-y-3">
              <h1 className="font-display text-4xl tracking-tight sm:text-5xl">
                Verifiez votre boite mail
              </h1>
              <p className="max-w-2xl text-sm leading-7 text-white/78 sm:text-base">
                Votre compte a bien ete cree. Pour activer votre espace Nova Assurances et acceder
                a vos devis, confirmez d&apos;abord votre adresse email.
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-5 pt-6">
          <div className="rounded-[24px] border border-[var(--line)] bg-white/82 p-5 text-sm leading-7 text-[var(--foreground)]">
            <p className="font-semibold">Etape suivante</p>
            <p className="mt-2">
              Ouvrez l&apos;email de confirmation que nous venons de vous envoyer
              {email ? ` a ${email}` : ""}, puis cliquez sur le lien d&apos;activation.
            </p>
            <p className="mt-2 text-[var(--muted)]">
              Une seule confirmation suffit. Ensuite, vous pourrez vous connecter normalement quand
              vous le souhaitez.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button asChild>
              <Link href="/connexion">Me connecter apres validation</Link>
            </Button>
            <Button asChild variant="secondary">
              <Link href="/">Retour a l&apos;accueil</Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
