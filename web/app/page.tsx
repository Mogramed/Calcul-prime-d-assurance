import { ArrowRight, Clock3, ShieldCheck, Sparkles, Waypoints } from "lucide-react";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

const highlights = [
  {
    title: "Une estimation claire",
    description: "La prime estimee est mise en avant avec un recapitulatif simple et lisible.",
    icon: Sparkles,
  },
  {
    title: "Disponible sur cet appareil",
    description: "Vos derniers devis restent accessibles ici pour etre relus ou modifies facilement.",
    icon: ShieldCheck,
  },
  {
    title: "Parcours guide",
    description: "Trois etapes simples suffisent pour arriver a une estimation de prime auto.",
    icon: Waypoints,
  },
];

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_400px]">
        <Card className="overflow-hidden">
          <CardHeader className="border-b border-[var(--line)] bg-[linear-gradient(135deg,rgba(14,26,42,0.98),rgba(26,40,60,0.92))] pb-8 text-white">
            <div className="space-y-4">
              <Badge className="border-white/20 bg-white/10 text-white">Nova Assurances</Badge>
              <div className="space-y-3">
                <h1 className="max-w-4xl font-display text-5xl leading-none tracking-tight sm:text-6xl">
                  Obtenez une estimation claire de votre assurance auto.
                </h1>
                <p className="max-w-2xl text-base leading-8 text-white/78">
                  Nova Assurances vous accompagne dans un parcours simple, rassurant et sans jargon
                  pour estimer votre prime auto en quelques minutes.
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-6 pt-6 lg:grid-cols-[1fr_auto]">
            <div className="space-y-4 text-sm leading-8 text-[var(--muted)]">
              <p>
                Un assistant en trois etapes vous guide pour renseigner votre profil, les conducteurs
                et le vehicule, puis affiche une estimation lisible de votre prime.
              </p>
              <p>
                Vous pouvez reprendre vos derniers devis sur cet appareil et les modifier sans avoir a
                tout ressaisir.
              </p>
            </div>
            <div className="flex flex-col gap-3">
              <Button asChild className="w-full sm:w-auto">
                <Link href="/devis">
                  Demarrer un devis
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="secondary" className="w-full sm:w-auto">
                <Link href="/mes-devis">Retrouver mes devis</Link>
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-4">
            <Badge className="w-fit">Parcours client</Badge>
            <h2 className="font-display text-3xl text-[var(--foreground)]">3 etapes pour aller a l&rsquo;essentiel</h2>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-[var(--muted)]">
            <div className="rounded-2xl border border-[var(--line)] bg-white/70 px-4 py-3">
              <strong className="text-[var(--foreground)]">1. Profil et formule</strong>
              <br />
              Choisissez votre formule, votre rythme de paiement et l&rsquo;usage du vehicule.
            </div>
            <div className="rounded-2xl border border-[var(--line)] bg-white/70 px-4 py-3">
              <strong className="text-[var(--foreground)]">2. Conducteurs</strong>
              <br />
              Renseignez les informations principales sur le conducteur et, si besoin, un second conducteur.
            </div>
            <div className="rounded-2xl border border-[var(--line)] bg-white/70 px-4 py-3">
              <strong className="text-[var(--foreground)]">3. Vehicule</strong>
              <br />
              Finalisez avec les caracteristiques du vehicule pour afficher votre estimation.
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        {highlights.map((item) => (
          <Card key={item.title}>
            <CardContent className="pt-6">
              <item.icon className="h-6 w-6 text-[var(--accent)]" />
              <h2 className="mt-4 font-display text-2xl text-[var(--foreground)]">{item.title}</h2>
              <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{item.description}</p>
            </CardContent>
          </Card>
        ))}
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardContent className="pt-6">
            <Clock3 className="h-6 w-6 text-[var(--accent)]" />
            <h2 className="mt-4 font-display text-2xl text-[var(--foreground)]">Rapide a remplir</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
              Le parcours est pense pour avancer sereinement, avec des champs guides et des choix simples.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <ShieldCheck className="h-6 w-6 text-[var(--accent)]" />
            <h2 className="mt-4 font-display text-2xl text-[var(--foreground)]">Rassurant et lisible</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
              Le resultat va droit au but, sans exposer d&rsquo;informations techniques inutiles.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <Sparkles className="h-6 w-6 text-[var(--accent)]" />
            <h2 className="mt-4 font-display text-2xl text-[var(--foreground)]">Pense pour revenir plus tard</h2>
            <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
              Vos derniers devis restent disponibles sur cet appareil pour reprendre facilement votre saisie.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
