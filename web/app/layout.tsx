import type { Metadata } from "next";
import { Bricolage_Grotesque, DM_Sans } from "next/font/google";
import Link from "next/link";

import { Providers } from "@/components/providers";
import { getCurrentSessionUser } from "@/lib/server/current-user";
import "./globals.css";

const sans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
});

const display = Bricolage_Grotesque({
  variable: "--font-bricolage",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Nova Assurances",
  description: "Estimation de prime auto claire et guidee pour vos clients.",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const currentUser = await getCurrentSessionUser();

  return (
    <html lang="fr" className={`${sans.variable} ${display.variable} h-full`}>
      <body className="min-h-full bg-[var(--background)] text-[var(--foreground)] antialiased">
        <Providers>
          <div className="relative min-h-screen overflow-hidden">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(199,152,104,0.24),transparent_32%),radial-gradient(circle_at_bottom_right,rgba(20,33,50,0.12),transparent_28%)]" />
            <div className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 pb-10 pt-5 sm:px-6 lg:px-8">
              <header className="mb-8 rounded-full border border-white/55 bg-white/65 px-5 py-3 shadow-[0_18px_60px_rgba(23,32,42,0.08)] backdrop-blur">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <Link href="/" className="flex items-center gap-3">
                    <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-[var(--navy)] font-display text-lg text-white">
                      NA
                    </span>
                    <span className="font-display text-2xl tracking-tight text-[var(--foreground)]">
                      Nova Assurances
                    </span>
                  </Link>
                  <nav className="flex flex-wrap items-center gap-2 text-sm font-medium text-[var(--muted)]">
                    <Link className="rounded-full px-4 py-2 hover:bg-white/70" href="/">
                      Accueil
                    </Link>
                    {currentUser ? (
                      <>
                        <Link className="rounded-full px-4 py-2 hover:bg-white/70" href="/devis">
                          Devis
                        </Link>
                        <Link className="rounded-full px-4 py-2 hover:bg-white/70" href="/mes-devis">
                          Mes devis
                        </Link>
                        <Link className="rounded-full px-4 py-2 hover:bg-white/70" href="/compte">
                          Mon compte
                        </Link>
                      </>
                    ) : (
                      <>
                        <Link className="rounded-full px-4 py-2 hover:bg-white/70" href="/connexion">
                          Connexion
                        </Link>
                        <Link className="rounded-full px-4 py-2 hover:bg-white/70" href="/inscription">
                          Inscription
                        </Link>
                      </>
                    )}
                    {currentUser?.role === "admin" ? (
                      <Link className="rounded-full px-4 py-2 hover:bg-white/70" href="/admin">
                        Admin
                      </Link>
                    ) : null}
                  </nav>
                </div>
              </header>

              <main className="flex-1">{children}</main>

              <footer className="mt-8 border-t border-white/40 px-2 pt-5 text-sm text-[var(--muted)]">
                <div className="grid gap-3 lg:grid-cols-[auto_1fr_auto] lg:items-start">
                  <p className="font-medium text-[var(--foreground)]">Nova Assurances</p>
                  <div className="space-y-1 leading-7">
                    <p>
                      Projet demonstrateur de tarification auto. Les estimations affichees sur ce site sont fournies a titre indicatif, sans valeur contractuelle, et ne constituent pas une offre d'assurance.
                    </p>
                    <p>
                      Responsable de publication : Mohamed Khaldi. Contact : khaldimohamedamine78@gmail.com.
                    </p>
                    <p>
                      Hebergement : Google Cloud Run. Donnees de devis reservees au fonctionnement du service et a la demonstration du projet.
                    </p>
                  </div>
                  <p className="text-left lg:text-right">Version de demonstration</p>
                </div>
              </footer>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
